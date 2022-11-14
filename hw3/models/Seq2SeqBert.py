import torch.nn as nn
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, AutoTokenizer


# Reference:
# https://huggingface.co/docs/transformers/model_doc/encoder-decoder
# https://huggingface.co/docs/transformers/model_doc/bert-generation

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

    def forward(self, episodes, action_labels, target_labels):
        """
        parameters:
        - episodes: [batch_size, seq_len]
        - action_labels: [batch_size, num_of_instruction_in_one_episode]
        - target_labels: [batch_size, num_of_instruction_in_one_episode]

        return: https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput
        action_outputs: transformers.modeling_outputs.Seq2SeqLMOutput
        target_outputs: transformers.modeling_outputs.Seq2SeqLMOutput
        """
        action_outputs = self.bert2bert(input_ids=episodes, decoder_input_ids=action_labels, labels=action_labels)
        target_outputs = self.bert2bert(input_ids=episodes, decoder_input_ids=target_labels, labels=target_labels)

        return action_outputs, target_outputs
