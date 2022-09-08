import json

import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import matplotlib.pyplot as plt

from utils import (
    get_device,
    build_tokenizer_table,
    build_output_tables,
    encode_data
)

from model import AlfredClassifier


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
        - len(vocab_to_index): total number of words
        - len(actions_to_index): total number of distinct actions
        - len(targets_to_index): total number of distinct targets
    """
    # ===================================================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.
    # ===================================================== #
    batch_size = args.batch_size
    data_file = args.in_data_fn

    # Load data from json file
    file = open(data_file)
    data = json.load(file)
    # Read in training data
    training_data = []
    validation_data = []
    for episode in data["train"]:
        for entry in episode:
            training_data.append(entry)

    # Read in validation data
    for episode in data["valid_seen"]:
        for entry in episode:
            validation_data.append(entry)

    file.close()

    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff, embedding_matrix = build_tokenizer_table(train=training_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train=training_data)

    # Encode the training and validation set inputs/outputs.
    train_instructions, train_labels = encode_data(training_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    val_instructions, val_labels = encode_data(validation_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    train_dataset = TensorDataset(torch.from_numpy(train_instructions), torch.from_numpy(train_labels))
    val_dataset = TensorDataset(torch.from_numpy(val_instructions), torch.from_numpy(val_labels))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, val_loader, len(vocab_to_index), len(actions_to_index), len(targets_to_index), embedding_matrix


def setup_model(args, n_vocab: int, n_actions: int, n_targets: int, embedding_matrix: list):
    """
    return:
        - model: AlfredClassifier
    """
    # ===================================================== #
    # Task: Initialize your model.
    # ===================================================== #
    n_embedding = 100
    n_hidden = 64
    n_hidden_layer = 1
    dropout_rate = 0.3
    model = AlfredClassifier(n_vocab=n_vocab, n_embedding=n_embedding, n_hidden=n_hidden, dropout_rate=dropout_rate, n_actions=n_actions, n_targets=n_targets, n_hidden_layer=n_hidden_layer)
    model.embedding_layer.weight.data.copy_(torch.as_tensor(embedding_matrix))
    return model


def setup_optimizer(args, model, device):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ===================================================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = nn.CrossEntropyLoss().to(device)
    target_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(params=model.parameters())

    return action_criterion, target_criterion, optimizer


def train_epoch(
        args,
        model,
        loader,
        optimizer,
        action_criterion,
        target_criterion,
        device,
        training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    model.train()
    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)
        actions_out = actions_out.squeeze(1)[:, -1]
        targets_out = targets_out.squeeze(1)[:, -1]

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out, labels[:, 0].long())
        target_loss = target_criterion(targets_out, labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    epoch_action_loss /= len(target_preds)
    epoch_target_loss /= len(target_preds)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
        args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()
    all_train_action_loss = []
    all_train_target_loss = []
    all_train_action_acc = []
    all_train_target_acc = []
    all_val_action_loss = []
    all_val_target_loss = []
    all_val_action_acc = []
    all_val_target_acc = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )
        all_train_action_loss.append(train_action_loss)
        all_train_target_loss.append(train_target_loss)
        all_train_action_acc.append(train_action_acc)
        all_train_target_acc.append(train_target_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target acc: {val_target_acc}"
            )
            all_val_action_loss.append(val_action_loss)
            all_val_target_loss.append(val_target_loss)
            all_val_action_acc.append(val_action_acc)
            all_val_target_acc.append(val_target_acc)

    # ===================================================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
    train_graph_x_axis_labels = [int(i) for i in range(1, args.num_epochs + 1)]
    val_graph_x_axis_labels = [int(i) for i in range(1, args.num_epochs + 1, args.val_every)]

    figure, ((train_loss_graph, train_acc_graph), (val_loss_graph, val_acc_graph)) = plt.subplots(2, 2)
    train_loss_graph.plot(train_graph_x_axis_labels, all_train_action_loss, label="Action")
    train_loss_graph.plot(train_graph_x_axis_labels, all_train_target_loss, label="Target")
    train_acc_graph.plot(train_graph_x_axis_labels, all_train_action_acc, label="Action")
    train_acc_graph.plot(train_graph_x_axis_labels, all_train_target_acc, label="Target")
    val_loss_graph.plot(val_graph_x_axis_labels, all_val_action_loss, label="Action")
    val_loss_graph.plot(val_graph_x_axis_labels, all_val_target_loss, label="Target")
    val_acc_graph.plot(val_graph_x_axis_labels, all_val_action_acc, label="Action")
    val_acc_graph.plot(val_graph_x_axis_labels, all_val_target_acc, label="Target")
    train_loss_graph.set_title("Training Loss")
    train_loss_graph.set_ylim(bottom=0)
    train_loss_graph.set_xlabel("Num of epochs")
    train_acc_graph.set_title("Training Accuracy")
    train_acc_graph.set_ylim(bottom=0)
    train_acc_graph.set_xlabel("Num of epochs")
    val_loss_graph.set_title("Validation Loss")
    val_loss_graph.set_ylim(bottom=0)
    val_loss_graph.set_xlabel("Num of epochs")
    val_acc_graph.set_title("Validation Accuracy")
    val_acc_graph.set_ylim(bottom=0)
    val_acc_graph.set_xlabel("Num of epochs")
    figure.legend(labels=["Action", "Target"], loc='upper left')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

    # Save result graph
    figure.savefig("result.png")


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, n_vocab, n_actions, n_targets, embedding_matrix = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, n_vocab, n_actions, n_targets, embedding_matrix).to(device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model, device)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
        print(f"val action loss: {val_action_loss} | val action acc: {val_action_acc}")
        print(f"val target loss: {val_target_loss} | val target acc: {val_target_acc}")
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )

    args = parser.parse_args()

    main(args)
