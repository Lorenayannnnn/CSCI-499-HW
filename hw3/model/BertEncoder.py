
import torch.nn as nn
from transformers import BertModel


# Reference:
# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel


class BertEncoder(nn.Module):
    def __init__(self,
                 model_name: str,
                 input_bos_token_id: int,
                 input_eos_token_id: int,
                 input_pad_token_id: int,
                 input_n_vocab: int):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name,
                                                 vocab_size=input_n_vocab,
                                                 bos_token_id=input_bos_token_id,
                                                 eos_token_id=input_eos_token_id,
                                                 pad_token_id=input_pad_token_id,
                                                 output_hidden_states=True,
                                                 ignore_mismatched_sizes=True)

    def forward(self, episodes, _):
        """
        parameters:
        - episodes: [batch_size, seq_len]

        return: https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        - encoder_hidden_outputs: [batch_size, seq_len, hidden_dim]
        - hidden: [1, batch_size, hidden_dim]
        """
        outputs = self.encoder(episodes)
        encoder_hidden_outputs = outputs.last_hidden_state
        hidden = outputs.pooler_output.unsqueeze(0)

        return encoder_hidden_outputs, hidden
