import re

import torch
import numpy as np
from collections import Counter


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
    inst_count = len(train)
    for entry in train:
        inst = preprocess_string(entry[0])
        padded_len = 2  # start/end
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

    word_to_embeddings = load_glove_word_embedding()
    embedding_matrix = np.zeros((len(vocab_to_index), 100))
    for i, word in enumerate(vocab_to_index):
        embedding_vector = word_to_embeddings.get(word)
        if embedding_vector is not None:
            # words not found in word_to_embeddings will be all-zeros.
            embedding_matrix[i+4] = embedding_vector
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
        embedding_matrix
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for entry in train:
        a = entry[1][0]
        t = entry[1][1]
        actions.add(a)
        targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


def encode_data(training_data: list, vocab_to_index: dict, actions_to_index: dict, targets_to_index: dict,
                seq_len: int):
    n_instructions = len(training_data)
    encoded_instructions = np.zeros((n_instructions, seq_len), dtype=np.int32)
    encoded_labels = np.zeros((n_instructions, 2), dtype=np.int32)

    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for (idx, entry) in enumerate(training_data):
        # TODO
        instruction = entry[0]
        if idx == 0:
            instruction += f" {training_data[idx + 1][0]}"
        elif idx == len(training_data) - 1:
            instruction = f"{training_data[idx - 1][0]} {instruction}"
        else:
            instruction = f"{training_data[idx + 1][0]}"

        processed_instruction = preprocess_string(entry[0])
        action = entry[1][0]
        target = entry[1][1]
        encoded_instructions[idx][0] = vocab_to_index["<start>"]
        jdx = 1
        for word in processed_instruction.split():
            if len(word) > 0:
                encoded_instructions[idx][jdx] = vocab_to_index[word] if word in vocab_to_index else vocab_to_index[
                    "<unk>"]
                n_unks += 1 if encoded_instructions[idx][jdx] == vocab_to_index["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        encoded_instructions[idx][jdx] = vocab_to_index["<end>"]
        encoded_labels[idx] = [actions_to_index[action], targets_to_index[target]]
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(vocab_to_index))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    return encoded_instructions, encoded_labels


def load_glove_word_embedding():
    """
    :return: dictionary from word to its embedding
    """
    word_to_embeddings = {}
    file = open('glove.6B.100d.txt', 'rb')
    for line in file:
        values = line.split()
        word = values[0].decode('UTF-8')
        coef = np.asarray(values[1:], dtype='float32')
        word_to_embeddings[word] = coef
    file.close()
    return word_to_embeddings

