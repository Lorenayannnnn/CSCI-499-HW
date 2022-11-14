
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    """

    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate):
        super(Encoder, self).__init__()
        self.n_vocab = n_vocab
        self.embedding_layer = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_dim)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_hidden_layer,
                            batch_first=True, dropout=dropout_rate)

    def forward(self, episodes, seq_lens):
        """
        - episodes: [batch_size, seq_len]
        - seq_lens: [batch_size] (store how long each episode instruction is)
        """

        # embedding_out: [batch_size, seq_len, embedding_dim]
        embedding_out = self.embedding_layer(episodes)

        # Ignore padding tokens
        embedding_out_packed = pack_padded_sequence(embedding_out, seq_lens, batch_first=True, enforce_sorted=False)

        # lstm_out: [batch_size, seq_len, hidden_dim]
        # hidden: [1*num_layers, batch_size, hidden_dim]
        # cell: [1*num_layers, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.LSTM(embedding_out_packed)

        padded_lstm_out, size = pad_packed_sequence(lstm_out, batch_first=True)

        return padded_lstm_out, hidden, cell


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    """

    def __init__(self, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets,
                 use_encoder_decoder_attention):
        super(Decoder, self).__init__()
        self.n_hidden_layer = n_hidden_layer
        self.action_embedding_layer = nn.Embedding(num_embeddings=n_actions, embedding_dim=embedding_dim)
        self.target_embedding_layer = nn.Embedding(num_embeddings=n_targets, embedding_dim=embedding_dim)
        self.use_encoder_decoder_attention = use_encoder_decoder_attention
        self.LSTM = nn.LSTM(input_size=embedding_dim*2, hidden_size=hidden_dim, num_layers=n_hidden_layer,
                            dropout=dropout_rate)
        self.action_fc = nn.Linear(in_features=hidden_dim, out_features=n_actions)
        self.target_fc = nn.Linear(in_features=hidden_dim, out_features=n_targets)
        self.attention = Attention(hidden_dim)

    def forward(self, x, hidden, cell, encoder_hidden_outputs):
        """
        - x: [2, batch_size] containing previously predicted pair of action and target
        - hidden: [1*num_layers, batch_size, hidden_dim]
        - cell: [1*num_layers, batch_size, hidden_dim]
        - encoder_hidden_outputs: [batch_size, seq_len, hidden_dim]
        """
        # embedding_out: [batch_size, embedding_dim]
        action_embedding_out = self.action_embedding_layer(x[0])
        target_embedding_out = self.target_embedding_layer(x[1])
        embedding_out = torch.concat((action_embedding_out, target_embedding_out), dim=1).unsqueeze(0)
        # lstm_out: [2 (action & target), batch_size, hidden_dim]
        # new_hidden: [1*num_layers, batch_size, hidden_dim]
        # new_cell: [1*num_layers, batch_size, hidden_dim]
        lstm_out, (new_hidden, new_cell) = self.LSTM(embedding_out, (hidden, cell))
        if self.use_encoder_decoder_attention:
            # Only use hidden state of the last layer
            lstm_out = self.attention(new_hidden[self.n_hidden_layer - 1], encoder_hidden_outputs).unsqueeze(0)
        # action_out: [batch_size, n_actions]
        action_output = self.action_fc(lstm_out).squeeze(0)
        # target_output: [batch_size, n_targets]
        target_output = self.target_fc(lstm_out).squeeze(0)

        return action_output, target_output, new_hidden, new_cell


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    """

    def __init__(self, n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets,
                 teacher_forcing, use_encoder_decoder_attention):
        super(EncoderDecoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.teacher_forcing = teacher_forcing
        self.use_encoder_decoder_attention = use_encoder_decoder_attention
        self.encoder = Encoder(n_vocab, embedding_dim, hidden_dim, n_hidden_layer, dropout_rate)
        self.decoder = Decoder(embedding_dim, hidden_dim, n_hidden_layer, dropout_rate, n_actions, n_targets,
                               self.use_encoder_decoder_attention)

    def forward(self, episodes, labels, seq_lens):
        """
        parameters:
        - episodes: [batch_size, seq_len]
        - labels: [batch_size, num_of_instruction_in_one_episode, 2(containing action and target)]
        - seq_lens [batch_size] (store how long each episode instruction is)

        return:
        - all_predicted_actions ([batch_size, instruction_num])
        - all_predicted_targets ([batch_size, instruction_num])
        - action_prob_dist ([batch_size, instruction_num, n_actions])
        - target_prob_dist ([batch_size, instruction_num, n_targets])
        """
        encoder_lstm_out, hidden, cell = self.encoder(episodes, seq_lens)
        batch_size = len(labels)
        instruction_num = len(labels[0])
        labels = torch.transpose(labels, 0, 1)

        # Store predicted distribution of action & target: [instruction_num, batch_size, n_actions/n_targets]
        action_prob_dist = torch.zeros((instruction_num, batch_size, self.n_actions))
        target_prob_dist = torch.zeros((instruction_num, batch_size, self.n_targets))
        # each time step has a row of action & a row of target for each sample in the batch
        all_predicted_actions = torch.zeros((instruction_num, batch_size))
        all_predicted_targets = torch.zeros((instruction_num, batch_size))
        # Corresponds to A_START and T_START tokens
        predicted_pairs = torch.zeros((2, batch_size), dtype=torch.long)
        for i in range(1, instruction_num):
            action_output, target_output, hidden, cell = self.decoder(predicted_pairs, hidden, cell, encoder_lstm_out)
            # Update predicted pair (depends on whether using teacher-forcing)
            predicted_action = torch.argmax(action_output, dim=1)
            predicted_target = torch.argmax(target_output, dim=1)
            predicted_pairs = torch.concat((predicted_action, predicted_target)).reshape(2, batch_size)
            # Store result
            action_prob_dist[i-1] = action_output
            target_prob_dist[i-1] = target_output
            all_predicted_actions[i-1] = predicted_action
            all_predicted_targets[i-1] = predicted_target
            # Use true labels if teacher_forcing
            predicted_pairs = torch.transpose(labels[i], 0, 1) if self.teacher_forcing else predicted_pairs

        return (
            torch.transpose(all_predicted_actions, 0, 1),
            torch.transpose(all_predicted_targets, 0, 1),
            torch.transpose(action_prob_dist, 0, 1),
            torch.transpose(target_prob_dist, 0, 1)
        )


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_score = nn.Linear(in_features=2 * hidden_dim, out_features=1)

    def forward(self, decoder_hidden, encoder_hidden_outputs):
        """
        - decoder_hidden: [batch_size, hidden_dim]
        - encoder_hidden_outputs: [batch_size, seq_en(global), hidden_dim]
        """
        batch_size = encoder_hidden_outputs.shape[0]
        seq_len = encoder_hidden_outputs.shape[1]

        # Concatenate decoder hidden state with each encoder hidden state
        repeated_decoder_hidden = torch.zeros((batch_size, seq_len, self.hidden_dim))
        for idx, hidden in enumerate(decoder_hidden):
            repeated_decoder_hidden[idx] = hidden.repeat(1, seq_len, 1)
        # concatenated_hidden: [batch_size, seq_len, 2*hidden_dim]
        concatenated_hidden = torch.cat((repeated_decoder_hidden, encoder_hidden_outputs), dim=2)
        attention_scores = self.attention_score(concatenated_hidden)
        weights = torch.nn.functional.softmax(attention_scores, dim=1)
        # [batch_size, seq_len]
        final_hidden = torch.bmm(encoder_hidden_outputs.transpose(1, 2), weights).squeeze(2)
        return final_hidden



