
from torch import nn

class SkipGramModel(nn.Module):
    """
    Define model for learning word embeddings (Skip-gram)
    """

    def __init__(self, n_vocab: int, n_embedding: int, context_window_len: int):
        super(SkipGramModel, self).__init__()
        self.n_vocab = n_vocab
        self.n_embedding = n_embedding
        self.context_window_len = context_window_len
        self.embedding_layer = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_embedding)
        self.fc = nn.Linear(in_features=n_embedding, out_features=context_window_len*2)


    def forward(self, input_token):
        embedding_out = self.embedding_layer(input_token)
        fc_out = self.fc(embedding_out)
        return fc_out
