import torch
from torch import nn


class AlfredClassifier(nn.Module):
    """
    Define model for classifying Alfred instructions
    """

    def __init__(self, n_vocab, n_embedding, n_hidden, dropout_rate, n_actions, n_targets, n_hidden_layer):
        super(AlfredClassifier, self).__init__()
        self.n_vocab = n_vocab
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_dropout = dropout_rate
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.n_hidden_layer = n_hidden_layer

        # embedding, LSTM, dropout, fc, activation
        self.embedding_layer = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embedding)
        self.LSTM = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, batch_first=True, num_layers=n_hidden_layer)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        # 2 independent prediction heads
        self.action_fc = nn.Linear(in_features=n_hidden, out_features=n_actions)
        self.target_fc = nn.Linear(in_features=n_hidden, out_features=n_targets)

        # Target takes in action
        # self.action_fc = nn.Linear(in_features=n_hidden, out_features=n_actions)
        # self.target_fc = nn.Linear(in_features=n_hidden + n_actions, out_features=n_targets)

        # Action takes in target
        # self.target_fc = nn.Linear(in_features=n_hidden, out_features=n_targets)
        # self.action_fc = nn.Linear(in_features=n_hidden + n_targets, out_features=n_actions)


    def forward(self, input_instructions):
        # permute input instructions
        embeddings = self.embedding_layer(input_instructions)
        lstm_output, _ = self.LSTM(embeddings)
        dropout_output = self.dropout_layer(lstm_output)

        # 2 independent prediction heads
        action_output = self.action_fc(dropout_output)
        target_output = self.target_fc(dropout_output)

        # Target takes in action
        # action_output = self.action_fc(dropout_output)
        # target_output = self.target_fc(torch.cat((dropout_output, action_output), dim=2))

        # Action takes in target
        # target_output = self.target_fc(dropout_output)
        # action_output = self.action_fc(torch.cat((dropout_output, target_output), dim=2))

        return action_output, target_output
