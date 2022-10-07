import json
import gensim
import tqdm
import numpy as np


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


def get_input_label_data_skip_gram(sentences: list, context_window_len: int, pad_token: int):
    """
    Parse all sentences and get input and labels (skip_gram)
    input: 1 token
    label: context word within the input context window length
    """
    input_data = []
    labels = []

    for sentence in sentences:
        for (i, token) in enumerate(sentence):
            input_data.append(token)
            context = []
            for index in range(i - context_window_len, i + context_window_len + 1):
                if index != i:
                    context.append(get_token(sentence, index, pad_token))
            labels.append(context)

    return input_data, labels
