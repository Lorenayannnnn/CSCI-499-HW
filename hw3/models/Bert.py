import torch.nn as nn
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, AutoTokenizer


class Seq2SeqBert(nn.Module):
    """
    Seq2Seq model based on Bert
    """
    def __init__(self, model_name: str, input_bos_token_id: int, input_eos_token_id: int, output_bos_token_id: int,
                 output_eos_token_id: int):
        super(Seq2SeqBert, self).__init__()
        self.encoder = BertGenerationEncoder.from_pretrained(model_name, bos_token_id=input_bos_token_id,
                                                             eos_token_id=input_eos_token_id)
        self.decoder = BertGenerationDecoder.from_pretrained(model_name, add_cross_attention=True,
                                                             is_decoder=True, bos_token_id=output_bos_token_id,
                                                             eos_token_id=output_eos_token_id)
        self.bert2bert = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, episodes, labels):
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
