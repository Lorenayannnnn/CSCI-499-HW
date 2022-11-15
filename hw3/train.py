import numpy as np
import tqdm
import torch
import argparse
import json
from torch.utils.data import TensorDataset, DataLoader
import os

from model.Seq2SeqBert import Seq2SeqBert
from utils import (
    get_device,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
    exact_match,
    encode_data,
    parse_action_target_labels,
    get_episode_seq_lens,
    percentage_match,
    output_acc_graph,
    output_loss_graph, get_labels_seq_lens
)

from model.EncoderDecoder import EncoderDecoder


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ===================================================== #
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
    # TODO
    len_cutoff = 100
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(training_data)

    train_episodes, train_labels = encode_data(training_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    val_episodes, val_labels = encode_data(validation_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    train_dataset = TensorDataset(torch.from_numpy(np.array(train_episodes)), torch.from_numpy(np.array(train_labels)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(val_episodes)), torch.from_numpy(np.array(val_labels)))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, len(vocab_to_index), len(actions_to_index), len(
        targets_to_index), len_cutoff, vocab_to_index, actions_to_index


def setup_model(args, device, n_vocab: int, n_actions: int, n_targets: int, vocab_to_index: dict, actions_to_index: dict):
    """
    return:
        - model: EncoderDecoder or Seq2SeqBert
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
    # model_name: str, input_bos_token_id: int, input_eos_token_id: int, output_bos_token_id: int,
    # output_eos_token_id
    if args.run_seq_2_seq_bert:
        model_name = "bert-base-uncased"
        model = Seq2SeqBert(
            model_name=model_name,
            input_bos_token_id=vocab_to_index['<start>'],
            input_eos_token_id=vocab_to_index['<end>'],
            output_bos_token_id=vocab_to_index['A_START'],
            output_eos_token_id=actions_to_index['A_STOP']
        )
    else:
        embedding_dim = 128
        hidden_dim = 64
        n_hidden_layer = 2
        dropout_rate = 0.3
        model = EncoderDecoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets,
                               args.teacher_forcing, args.encoder_decoder_attention)
        # model = torch.load(os.path.join(args.outputs_dir, args.model_output_filename))
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=2).to(device)
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

    epoch_action_loss = 0.0
    epoch_target_loss = 0.0
    epoch_action_exact_match_acc = 0.0
    epoch_action_prefix_match_acc = 0.0
    epoch_action_num_of_match_acc = 0.0
    epoch_target_exact_match_acc = 0.0
    epoch_target_prefix_match_acc = 0.0
    epoch_target_num_of_match_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for inputs, labels in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        action_labels, target_labels = parse_action_target_labels(labels)
        action_labels, target_labels = action_labels.long().to(device), target_labels.long().to(device)
        seq_lens = get_episode_seq_lens(inputs).to(device)
        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        all_predicted_actions, all_predicted_targets, action_prob_dist, target_prob_dist = model(inputs, labels,
                                                                                                 seq_lens,
                                                                                                 teacher_forcing=training)
        action_prob_dist = torch.transpose(action_prob_dist, 1, 2)
        target_prob_dist = torch.transpose(target_prob_dist, 1, 2)
        action_loss = criterion(action_prob_dist, action_labels)
        target_loss = criterion(target_prob_dist, target_labels)

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print("all_predicted_actions:", all_predicted_actions)
        # print("action_labels:", action_labels)
        labels_lens = get_labels_seq_lens(labels)
        action_exact_match_score = exact_match(all_predicted_actions, action_labels)
        action_prefix_match_score = prefix_match(all_predicted_actions, action_labels, labels_lens)
        action_num_of_match_score = percentage_match(all_predicted_actions, action_labels)
        target_exact_match_score = exact_match(all_predicted_targets, target_labels)
        target_prefix_match_score = prefix_match(all_predicted_targets, target_labels, labels_lens)
        target_num_of_match_score = percentage_match(all_predicted_targets, target_labels)

        epoch_action_loss += action_loss
        epoch_action_exact_match_acc += action_exact_match_score
        epoch_action_prefix_match_acc += action_prefix_match_score
        epoch_action_num_of_match_acc += action_num_of_match_score
        epoch_target_loss += target_loss
        epoch_target_exact_match_acc += target_exact_match_score
        epoch_target_prefix_match_acc += target_prefix_match_score
        epoch_target_num_of_match_acc += target_num_of_match_score

    epoch_action_loss /= len(loader)
    epoch_target_loss /= len(loader)
    epoch_action_exact_match_acc /= len(loader)
    epoch_action_prefix_match_acc /= len(loader)
    epoch_action_num_of_match_acc /= len(loader)
    epoch_target_exact_match_acc /= len(loader)
    epoch_target_prefix_match_acc /= len(loader)
    epoch_target_num_of_match_acc /= len(loader)

    return (
        epoch_action_loss,
        epoch_target_loss,
        epoch_action_exact_match_acc,
        epoch_action_prefix_match_acc,
        epoch_action_num_of_match_acc,
        epoch_target_exact_match_acc,
        epoch_target_prefix_match_acc,
        epoch_target_num_of_match_acc
    )


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_action_num_of_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc, val_target_num_of_match_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_action_num_of_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc, val_target_num_of_match_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation

    all_train_action_exact_match_acc = []
    all_train_action_prefix_match_acc = []
    all_train_action_num_of_match_acc = []
    all_train_action_loss = []

    all_train_target_exact_match_acc = []
    all_train_target_prefix_match_acc = []
    all_train_target_loss = []
    all_train_target_num_of_match_acc = []

    all_val_action_exact_match_acc = []
    all_val_action_prefix_match_acc = []
    all_val_action_num_of_match_acc = []
    all_val_action_loss = []

    all_val_target_exact_match_acc = []
    all_val_target_prefix_match_acc = []
    all_val_target_num_of_match_acc = []
    all_val_target_loss = []

    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        train_action_loss, train_target_loss, train_action_exact_match_acc, train_action_prefix_match_acc, train_action_num_of_match_acc, train_target_exact_match_acc, train_target_prefix_match_acc, train_target_num_of_match_acc = 0, 0, 0, 0, 0, 0, 0, 0
        if args.run_seq_2_seq_bert:
            pass
        else:
            # returns loss for action and target prediction and accuracy
            train_action_loss, train_target_loss, train_action_exact_match_acc, train_action_prefix_match_acc, train_action_num_of_match_acc, train_target_exact_match_acc, train_target_prefix_match_acc, train_target_num_of_match_acc = train_epoch(
                args,
                model,
                loaders["train"],
                optimizer,
                criterion,
                device,
            )
        # some logging
        print("-------- Action --------")
        print(
            f"train_loss: {train_action_loss} | train_exact_match_acc: {train_action_exact_match_acc} | train_prefix_match_acc: {train_action_prefix_match_acc} | train_num_of_match_acc: {train_action_num_of_match_acc}")
        all_train_action_loss.append(train_action_loss)
        all_train_action_exact_match_acc.append(train_action_exact_match_acc)
        all_train_action_prefix_match_acc.append(train_action_prefix_match_acc)
        all_train_action_num_of_match_acc.append(train_action_num_of_match_acc)
        print("-------- Target --------")
        print(
            f"train_loss: {train_target_loss} | train_exact_match_acc: {train_target_exact_match_acc} | train_prefix_match_acc: {train_target_prefix_match_acc} | train_num_of_match_acc: {train_target_num_of_match_acc}")
        all_train_target_loss.append(train_target_loss)
        all_train_target_exact_match_acc.append(train_target_exact_match_acc)
        all_train_target_prefix_match_acc.append(train_target_prefix_match_acc)
        all_train_target_num_of_match_acc.append(train_target_num_of_match_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_action_num_of_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc, val_target_num_of_match_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print("-------- Action --------")
            print(
                f"val_loss: {val_action_loss} | val_exact_match_acc acc: {val_action_exact_match_acc} | val_prefix_match_acc: {val_action_prefix_match_acc} | val_num_of_match_acc: {val_action_num_of_match_acc}")
            all_val_action_loss.append(val_action_loss)
            all_val_action_exact_match_acc.append(val_action_exact_match_acc)
            all_val_action_prefix_match_acc.append(val_action_prefix_match_acc)
            all_val_action_num_of_match_acc.append(val_action_num_of_match_acc)
            print("-------- Target --------")
            print(
                f"val_loss: {val_target_loss} | val_exact_match_acc acc: {val_target_exact_match_acc} | val_prefix_match_acc: {val_target_prefix_match_acc} | val_num_of_match_acc: {val_target_num_of_match_acc}")
            all_val_target_loss.append(val_target_loss)
            all_val_target_exact_match_acc.append(val_target_exact_match_acc)
            all_val_target_prefix_match_acc.append(val_target_prefix_match_acc)
            all_val_target_num_of_match_acc.append(val_target_num_of_match_acc)

            # Save model
            ckpt_file = os.path.join(args.outputs_dir, args.model_output_filename)
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

    # ===================================================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy, 3) validation loss, 4) validation accuracy
    # ===================================================== #
    output_loss_graph(args, "training_loss.png", all_train_action_loss, all_train_target_loss, "Training Loss", False)
    output_acc_graph(args, "training_action_accuracy.png", all_train_action_exact_match_acc,
                     all_train_action_prefix_match_acc, all_train_action_num_of_match_acc, "Action Accuracy(Training)",
                     False)
    output_acc_graph(args, "training_target_accuracy.png", all_train_target_exact_match_acc,
                     all_train_target_prefix_match_acc, all_train_target_num_of_match_acc, "Target Accuracy(Training)",
                     False)
    output_loss_graph(args, "validation_loss.png", all_val_action_loss, all_val_target_loss, "Validation Loss", True)
    output_acc_graph(args, "validation_action_accuracy.png", all_val_action_exact_match_acc, all_val_action_prefix_match_acc,
                     all_val_action_num_of_match_acc, "Action Accuracy(Validation)", True)
    output_acc_graph(args, "validation_target_accuracy.png", all_val_target_exact_match_acc, all_val_target_prefix_match_acc,
                     all_val_target_num_of_match_acc, "Target Accuracy(Validation)", True)


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, n_vocab, n_actions, n_targets, seq_len, vocab_to_index, actions_to_index = setup_dataloader(
        args)

    print("n_vocab:", n_vocab, "n_actions", n_actions, "n_targets", n_targets)

    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, n_vocab, n_actions, n_targets, vocab_to_index, actions_to_index).to(device)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model, device)

    if args.eval:
        val_action_loss, val_target_loss, val_action_exact_match_acc, val_action_prefix_match_acc, val_action_num_of_match_acc, val_target_exact_match_acc, val_target_prefix_match_acc, val_target_num_of_match_acc = validate(
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
        "--outputs_dir", type=str, help="where to save outputs"
    )
    parser.add_argument(
        "--model_output_filename", type=str, help="output filename of the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument(
        "--teacher_forcing", type=bool, default=False, help="whether use teacher_forcing"
    )
    parser.add_argument(
        "--encoder_decoder_attention", type=bool, default=False, help="whether use encoder decoder attention"
    )
    parser.add_argument(
        "--run_seq_2_seq_bert", type=bool, default=False, help="run seq2seq bert model"
    )

    args = parser.parse_args()

    main(args)
