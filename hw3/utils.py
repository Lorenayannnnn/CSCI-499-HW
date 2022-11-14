import os
import re
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        padded_len = 2  # start/end
        for inst, _ in episode:
            inst = preprocess_string(inst)
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
        padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    # Save space for START, STOP, PAD
    actions_to_index = {a: i + 4 for i, a in enumerate(actions)}
    targets_to_index = {t: i + 4 for i, t in enumerate(targets)}
    actions_to_index["A_START"] = 0
    targets_to_index["T_START"] = 0
    actions_to_index["A_STOP"] = 1
    targets_to_index["T_STOP"] = 1
    actions_to_index["A_PAD"] = 2
    targets_to_index["T_PAD"] = 2
    actions_to_index["A_UNK"] = 3
    targets_to_index["T_UNK"] = 3
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


def prefix_match(predicted_labels, gt_labels, labels_len):
    """
    predicted_labels/gt_labels: [batch_size, instruction_num]
    labels_len (how long each output instruction sequence is (exclude the padding): [batch_size]
    """
    # predicted_labels: [batch_size, seq_length, 2]
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1
    batch_size = len(gt_labels)
    seq_len = len(gt_labels[0])
    pm = 0.0
    for i in range(batch_size):
        j = 0
        for j in range(1, seq_len):
            if predicted_labels[i][j] != gt_labels[i][j] or (predicted_labels[i][j] == 1 and gt_labels[i][j] == 1):
                break
        pm += j / labels_len[i] if j <= labels_len[i] else j / seq_len

    return pm/batch_size


def exact_match(predicted_labels, gt_labels):
    """
    params dim: [batch_size, instruction_num]
    """
    """
    average # of exact match of 1 batch
    """
    batch_size = len(gt_labels)
    seq_len = len(gt_labels[0])
    em = 0.0
    for i in range(batch_size):
        same = True
        for j in range(1, seq_len):
            if predicted_labels[i][j] != gt_labels[i][j]:
                same = False
                break
            elif predicted_labels[i][j] == 1 and gt_labels[i][j] == 1:
                # STOP token
                break
        em += 1 if same else 0
    return em/batch_size


def percentage_match(predicted_labels, gt_labels):
    """
    params dim: [batch_size, instruction_num]
    """
    batch_size = len(gt_labels)
    seq_len = len(gt_labels[0])
    total_match_score = 0.0
    for i in range(batch_size):
        match_score = 0
        j = 0
        for j in range(1, seq_len):
            if predicted_labels[i][j] == gt_labels[i][j]:
                match_score += 1
            if predicted_labels[i][j] == 1 and gt_labels[i][j] == 1:
                # STOP token
                break
        total_match_score += match_score / j
    return total_match_score/batch_size


def encode_data(training_data: list, vocab_to_index: dict, actions_to_index: dict, targets_to_index: dict,
                seq_len: int):
    """
    training_data: list of list of (instructions and corresponding pairs of targets & actions)
    """
    n_episodes = len(training_data)
    encoded_episodes = np.zeros((n_episodes, seq_len), dtype=np.int32)
    # Store pairs of action index and target index of instructions of all episodes
    encoded_labels = []

    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    # Maximum number of action-target pairs of 1 episode among all
    max_action_target_pair_len = 0
    for (idx, episode) in enumerate(training_data):
        episode_labels = [[actions_to_index["A_START"], targets_to_index["T_START"]]]
        encoded_episodes[idx][0] = vocab_to_index["<start>"]
        jdx = 1
        for entry in episode:
            processed_instruction = preprocess_string(entry[0])
            action = entry[1][0]
            target = entry[1][1]

            for word in processed_instruction.split():
                if len(word) > 0:
                    encoded_episodes[idx][jdx] = vocab_to_index[word] if word in vocab_to_index else vocab_to_index[
                        "<unk>"]
                    n_unks += 1 if encoded_episodes[idx][jdx] == vocab_to_index["<unk>"] else 0
                    n_tks += 1
                    jdx += 1
                    if jdx == seq_len - 1:
                        break
            episode_labels.append([
                actions_to_index[action] if action in actions_to_index else actions_to_index["A_UNK"],
                targets_to_index[target] if target in targets_to_index else targets_to_index["T_UNK"]
            ])
            if jdx == seq_len - 1:
                n_early_cutoff += 1
                break
        encoded_episodes[idx][jdx] = vocab_to_index["<end>"]
        # Append STOP indicating the end of an episode
        episode_labels.append([actions_to_index["A_STOP"], targets_to_index["T_STOP"]])
        encoded_labels.append(episode_labels)
        max_action_target_pair_len = max(max_action_target_pair_len, len(episode_labels))

    # "Pad" all encoded_labels to max_action_target_pair_len with
    # [actions_to_index["A_STOP"], targets_to_index["T_STOP"]]
    for index, episode_label in enumerate(encoded_labels):
        encoded_labels[index].extend([[actions_to_index["A_PAD"], targets_to_index["T_PAD"]]] * (
                    max_action_target_pair_len - len(episode_label)))

    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(vocab_to_index))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    return encoded_episodes, encoded_labels


def parse_action_target_labels(labels):
    """
    label: [batch_size, instruction_num, 2]

    return:
    - action_labels: [batch_size, instruction_num]
    - action_labels: [batch_size, instruction_num]
    """
    action_labels = torch.zeros((len(labels), len(labels[0])))
    target_labels = torch.zeros((len(labels), len(labels[0])))
    for idx, label in enumerate(labels):
        action_labels[idx-1] = label[:, 0]
        target_labels[idx-1] = label[:, 1]

    return action_labels, target_labels


def get_episode_seq_lens(batch_input):
    batch_size = len(batch_input)
    batch_seq_lens = torch.zeros(batch_size)
    for idx, i_input in enumerate(batch_input):
        batch_seq_lens[idx] = np.where(i_input == 0)[0][0] if len(np.where(i_input == 0)[0]) > 0 else len(i_input)
    return batch_seq_lens


def get_labels_seq_lens(batch_labels):
    batch_size = len(batch_labels)
    batch_seq_lens = np.zeros(batch_size, dtype=int)
    for idx, i_label in enumerate(batch_labels):
        i_label = torch.transpose(i_label, 0, 1)
        batch_seq_lens[idx] = np.where(i_label[0] == 1)[0] if len(np.where(i_label[0] == 1)[0]) > 0 else len(i_label[0])
    return batch_seq_lens


def output_loss_graph(args, output_file_name: str, action_loss_data: list, target_loss_data: list, graph_title: str,
                      is_val: bool):
    x_axis_data = [i for i in range(0, args.num_epochs, args.val_every)] if is_val else [i for i in range(args.num_epochs)]

    figure, ax = plt.subplots()
    action_loss_data = [loss.detach() for loss in action_loss_data]
    target_loss_data = [loss.detach() for loss in target_loss_data]
    ax.plot(x_axis_data, action_loss_data, label="Action")
    ax.plot(x_axis_data, target_loss_data, label="Target")
    ax.set_title(graph_title)
    ax.set_xlabel("Num of epochs")
    ax.legend()

    figure.show()
    figure.savefig(os.path.join(args.outputs_dir, output_file_name))


def output_acc_graph(args, output_file_name: str, exact_match_data: list, prefix_match_data: list,
                     percentage_match: list, graph_title: str, is_val: bool):
    x_axis_data = [i for i in range(0, args.num_epochs, args.val_every)] if is_val else [i for i in range(args.num_epochs)]

    figure, ax = plt.subplots()
    ax.plot(x_axis_data, exact_match_data, label="Exact Match")
    ax.plot(x_axis_data, prefix_match_data, label="Prefix Match")
    ax.plot(x_axis_data, percentage_match, label="Percentage Match")
    ax.set_title(graph_title)
    ax.set_xlabel("Num of epochs")
    ax.legend()

    figure.show()
    figure.savefig(os.path.join(args.outputs_dir, output_file_name))

