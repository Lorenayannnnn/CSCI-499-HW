# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO check
    """

    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate):
        super(Encoder, self).__init__()
        self.n_vocab = n_vocab
        self.embedding_layer = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_dim)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_hidden_layer,
                            batch_first=True, dropout=dropout_rate)

    def forward(self, episodes):
        # episodes: [batch_size, seq_len]

        # embedding_out: [batch_size, seq_len, embedding_dim]
        embedding_out = self.embedding_layer(episodes)

        # hidden: [1*num_layers, batch_size, hidden_dim]
        # cell: [1*num_layers, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.LSTM(embedding_out)

        return hidden, cell


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO check
    """

    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets):
        super(Decoder, self).__init__()
        self.n_vocab = n_vocab
        self.embedding_layer = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_dim)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_hidden_layer,
                            dropout=dropout_rate, batch_first=True)
        self.action_fc = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.target_fc = nn.Linear(in_features=hidden_dim, out_features=n_targets)

    def forward(self, x, hidden, cell):
        """
        - x: [batch_size, 2] containing previously predicted pair of action and target
        - hidden: [1*num_layers, batch_size, hidden_dim]
        - cell: [1*num_layers, batch_size, hidden_dim]
        """
        embedding_out = self.embedding_layer(x)
        lstm_out, (new_hidden, new_cell) = self.LSTM(embedding_out, hidden, cell)
        action_output = self.action_fc(lstm_out)
        target_output = self.target_fc(lstm_out)

        return action_output, target_output, new_hidden, new_cell


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets):
        super(EncoderDecoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.encoder = Encoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate)
        self.decoder = Decoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets)


    def forward(self, episodes, labels):
        """
        episodes: [batch_size, seq_len]
        labels: [batch_size, num_of_instruction_in_one_episode, 2(containing action and target)]
        """

