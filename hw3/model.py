# IMPLEMENT YOUR MODEL CLASS HERE
import numpy as np
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
                            dropout=dropout_rate)
        self.action_fc = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.target_fc = nn.Linear(in_features=hidden_dim, out_features=n_targets)

    def forward(self, x, hidden, cell):
        """
        - x: [2, batch_size] containing previously predicted pair of action and target
        - hidden: [1*num_layers, batch_size, hidden_dim]
        - cell: [1*num_layers, batch_size, hidden_dim]
        """
        # embedding_out: [2, batch_size, embedding_dim]
        embedding_out = self.embedding_layer(x)
        # lstm_out: [2 (action & target), batch_size, hidden_dim]
        # new_hidden: [1*num_layers, batch_size, hidden_dim]
        # new_cell: [1*num_layers, batch_size, hidden_dim]
        lstm_out, (new_hidden, new_cell) = self.LSTM(embedding_out, hidden, cell)
        # action_out: [batch_size, n_actions]
        action_output = self.action_fc(lstm_out[0])
        # target_output: [batch_size, n_targets]
        target_output = self.target_fc(lstm_out[1])

        return action_output, target_output, new_hidden, new_cell


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets,
                 teacher_forcing):
        super(EncoderDecoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.teacher_forcing = teacher_forcing
        self.encoder = Encoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate)
        self.decoder = Decoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets)


    def forward(self, episodes, labels):
        """
        episodes: [batch_size, seq_len]
        labels: [batch_size, num_of_instruction_in_one_episode, 2(containing action and target)]
        """
        hidden, cell = self.encoder(episodes)
        batch_size = len(labels)
        instruction_num = len(labels[0])
        labels = np.transpose(labels, axes=[1, 0, 2])

        # Store predicted distribution of action & target: [instruction_num, batch_size, n_actions/n_targets]
        action_prob_dist = []
        target_prob_dist = []
        all_predicted_pairs = []
        # Corresponds to A_START and T_START tokens
        predicted_pairs = np.zeros((2, batch_size))
        for i in range(1, instruction_num):
            action_output, target_output, hidden, cell = self.decoder(predicted_pairs, hidden, cell)
            # Store result
            action_prob_dist.append(action_output)
            target_prob_dist.append(target_output)
            # Update predicted pair (depends on whether using teacher-forcing)
            predicted_action = np.argmax(action_output, axis=1)
            predicted_target = np.argmax(target_output, axis=1)
            predicted_pairs = [predicted_action, predicted_target]
            all_predicted_pairs.append(predicted_pairs)
            # Use true labels if teacher_forcing
            predicted_pairs = np.transpose(labels[i - 1]) if self.teacher_forcing else np.array(predicted_pairs)

        return np.transpose(np.array(all_predicted_pairs), axes=[2, 0, 1]), action_prob_dist, target_prob_dist
