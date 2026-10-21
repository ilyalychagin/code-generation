from transformers import T5ForConditionalGeneration, T5Config
import torch.nn as nn

class TestModel(nn.Module):

    def __init__(self, vocab_size=None, pad_token_id=None, eos_token_id=None, decoder_start_token_id=None):
        super().__init__()
        self.config = T5Config.from_pretrained('google-t5/t5-small')
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
        if pad_token_id is not None:
            self.config.pad_token_id = pad_token_id
        if eos_token_id is not None:
            self.config.eos_token_id = eos_token_id
        if decoder_start_token_id is not None:
            self.config.decoder_start_token_id = decoder_start_token_id

        self.model = T5ForConditionalGeneration(self.config)
    
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)