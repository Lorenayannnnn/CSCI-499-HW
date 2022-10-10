import json
import gensim
import numpy
import tqdm
import numpy as np
import torch
import pandas as pd


def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with gensim.utils.smart_open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embed.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )


def get_token(sentence: list, index: int, pad_token: int):
    """
    get token at input index of the sentence
    """
    return pad_token if (index < 0 or index >= len(sentence)) else sentence[index]


def create_train_val_splits(all_sentences: list, prop_train=0.7):
    """
    Split all input sentences to train and validation
    Reference: code snippet from coding activity 3
    """
    train_sentences = []
    val_sentences = []
    train_idxs = np.random.choice(list(range(len(all_sentences))), size=int(len(all_sentences) * prop_train + 0.5),
                                  replace=False)
    train_sentences.extend([all_sentences[idx] for idx in range(len(all_sentences)) if idx in train_idxs])
    val_sentences.extend([all_sentences[idx] for idx in range(len(all_sentences)) if idx not in train_idxs])
    return train_sentences, val_sentences


def get_input_label_data_skip_gram(sentences: list, context_window_len: int, pad_token: int, n_vocab: int):
    """
    Parse all sentences and get input and labels (skip_gram)
    token -> context word within the input context window length
    """
    context_list = []
    token_list = []

    for (sentence_idx, sentence) in enumerate(sentences):
        if sentence_idx % 10000 == 0:
            print(f"Parsed {sentence_idx}/{len(sentences)}")
        for (i, token) in enumerate(sentence):
            if token == 0:
                break
            token_list.append(token)
            context = [0] * n_vocab
            bound = int(context_window_len / 2)

            for index in range(i - bound, i + bound + 1):
                if index != i:
                    context[get_token(sentence, index, pad_token)] = 1
            context_list.append(context)

    return numpy.array(token_list), numpy.array(context_list)


def get_input_label_data_cbow(sentences: list, context_window_len: int, pad_token: int, lens):
    """
    Parse all sentences and get input and labels (skip_gram)
    context -> token
    """
    context_list = []
    tokens = []

    for (sentence_idx, sentence) in enumerate(sentences):
        if sentence_idx % 10000 == 0:
            print(f"Parsed {sentence_idx}/{len(sentences)}")
        for (i, token) in enumerate(sentence):
            if i >= lens[i][0]:
                break
            # elif token == 3:
            #     continue
            tokens.append(token)

            context = []
            bound = int(context_window_len / 2)
            for index in range(i - bound, i + bound + 1):
                if index != i:
                    context.append(get_token(sentence, index, pad_token))
            context_list.append(context)

    return numpy.array(context_list), numpy.array(tokens)


def get_train_val_dataset():
    train_df = pd.read_pickle("train.pkl")
    val_df = pd.read_pickle("val.pkl")

    # Read in data from local pickle file
    x_train = train_df["input_data"].values.tolist()
    y_train = train_df["labels"].values.tolist()
    x_val = val_df["input_data"].values.tolist()
    y_val = val_df["labels"].values.tolist()

    return x_train, y_train, x_val, y_val


def get_device(force_cpu, status=True):
    # Reference: from hw1

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


# TODO
def parse_skipgram_preds(prediction, context_window_len: int):
    index_list = [np.argpartition(indexes.detach().numpy(), -context_window_len)[-context_window_len::] for indexes in prediction]
    parsed_preds = []
    for i, a in enumerate(prediction):
        result = np.zeros(len(a))
        for index in index_list[i]:
            result[index] = 1
        parsed_preds.append(result)

    return torch.tensor(numpy.array(parsed_preds))
    # index_list = np.array([np.argpartition(indexes.detach().numpy(), -context_window_len)[-context_window_len::] for
    #               indexes in prediction])
    # index_list.sort()
    # return index_list