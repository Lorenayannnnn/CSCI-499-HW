import numpy as np
import tqdm
import torch
import argparse
import json
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    get_device,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
    exact_match,
    encode_data,
    parse_action_target_labels,
    output_result_figure
)

from model import EncoderDecoder


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CHECK ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    batch_size = args.batch_size
    data_file = args.in_data_fn

    # Load data from json file
    file = open(data_file)
    data = json.load(file)
    # Read in training and validation data
    training_data = [episode for episode in data["train"]]
    validation_data = [episode for episode in data["valid_seen"]]

    file.close()

    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(training_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(training_data)

    train_episodes, train_labels = encode_data(training_data, vocab_to_index, actions_to_index, targets_to_index,
                                               len_cutoff)
    val_episodes, val_labels = encode_data(validation_data, vocab_to_index, actions_to_index, targets_to_index,
                                           len_cutoff)
    train_dataset = TensorDataset(torch.from_numpy(np.array(train_episodes)), torch.from_numpy(np.array(train_labels)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(val_episodes)), torch.from_numpy(np.array(val_labels)))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, len(vocab_to_index), len(actions_to_index), len(targets_to_index)


def setup_model(args, device, n_vocab: int, n_actions: int, n_targets: int):
    """
    return:
        - model: EncoderDecoder
    """
    # ===================================================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    embedding_dim = 128
    hidden_dim = 64
    n_hidden_layer = 2
    dropout_rate = 0.3
    model = EncoderDecoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets,
                           args.teacher_forcing)
    return model


def setup_optimizer(args, model, device):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ===================================================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss(ignore_index=1).to(device)
    optimizer = torch.optim.Adam(params=model.parameters())

    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.

    --> Implemented in the EncoderDecoder model
    """

    epoch_loss = 0.0
    epoch_exact_match_acc = 0.0
    epoch_prefix_match_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        action_labels, target_labels = parse_action_target_labels(labels)
        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        all_predicted_pairs, action_prob_dist, target_prob_dist = model(inputs, labels)

        action_loss = criterion(action_prob_dist, action_labels)
        target_loss = criterion(target_prob_dist, target_labels)

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        exact_match_score = exact_match(all_predicted_pairs, labels)
        prefix_match_score = prefix_match(all_predicted_pairs, labels)

        # logging
        epoch_loss += loss.item()
        epoch_exact_match_acc += exact_match_score
        epoch_prefix_match_acc += prefix_match_score

    epoch_loss /= len(loader)
    epoch_exact_match_acc /= len(loader)
    epoch_prefix_match_acc /= len(loader)

    return epoch_loss, epoch_exact_match_acc, epoch_prefix_match_acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_exact_match_acc, val_prefix_match_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_exact_match_acc, val_prefix_match_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation

    all_train_exact_match_acc = []
    all_train_prefix_match_acc = []
    all_train_loss = []
    all_val_exact_match_acc = []
    all_val_prefix_match_acc = []
    all_val_loss = []

    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_exact_match_acc, train_prefix_match_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        print(
            f"train loss : {train_loss} | train_exact_match_acc: {train_exact_match_acc} | train_prefix_match_acc: {train_prefix_match_acc}")
        all_train_loss.append(train_loss)
        all_train_exact_match_acc.append(train_exact_match_acc)
        all_train_prefix_match_acc.append(train_prefix_match_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_exact_match_acc, val_prefix_match_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(
                f"val loss : {val_loss} | val_exact_match_acc acc: {val_exact_match_acc} | val_prefix_match_acc: {val_prefix_match_acc}")
            all_val_loss.append(val_loss)
            all_val_exact_match_acc.append(val_exact_match_acc)
            all_val_prefix_match_acc.append(val_prefix_match_acc)

    # ===================================================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy, 3) validation loss, 4) validation accuracy
    # ===================================================== #
    output_result_figure(args, "output_graphs/training_loss.png", all_train_loss, "Training Loss", False)
    output_result_figure(args, "output_graphs/training_acc(exact_match).png", all_train_exact_match_acc,
                         "Training Accuracy (exact match)", False)
    output_result_figure(args, "output_graphs/training_acc(prefix_match).png", all_train_prefix_match_acc,
                         "Training Accuracy (prefix match)", False)
    output_result_figure(args, "output_graphs/validation_loss.png", all_val_loss, "Validation Loss", True)
    output_result_figure(args, "output_graphs/validation_acc(exact_match).png", all_val_exact_match_acc,
                         "Validation Training Accuracy (exact match)", True)
    output_result_figure(args, "output_graphs/validation_acc(prefix_match).png", all_val_prefix_match_acc,
                         "Validation Training Accuracy (prefix match)", True)

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    # train_loader, val_loader, maps = setup_dataloader(args)
    train_loader, val_loader, n_vocab, n_actions, n_targets = setup_dataloader(args)

    print("n_vocab:", n_vocab)
    print("n_actions:", n_actions)
    print("n_targets:", n_targets)

    for input, label in train_loader:
        print(label)
        return

    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, n_vocab, n_actions, n_targets)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model, device)

    if args.eval:
        val_loss, val_exact_match_acc, val_prefix_match_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument(
        "--teacher_forcing", default=False, help="whether use teacher_forcing"
    )

    args = parser.parse_args()

    main(args)
