from transformers import T5ForConditionalGeneration, T5Config
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
import torch.nn as nn

class TestModel(nn.Module):

    def __init__(
        self, vocab_size=None, pad_token_id=None, eos_token_id=None,
        decoder_start_token_id=None, num_encoder_layers=6, num_decoder_layers=6,
        d_ff=2048, feed_forward_proj="relu",
    ):
        super().__init__()
        model_config = T5Config.from_pretrained('google-t5/t5-small').to_dict()
        model_config.update(d_ff=d_ff, feed_forward_proj=feed_forward_proj)
        model_config.pop("dense_act_fn", None)
        model_config.pop("is_gated_act", None)
        self.config = T5Config.from_dict(model_config)
        self.config.num_layers = num_encoder_layers
        self.config.num_decoder_layers = num_decoder_layers
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
        if self.config.num_layers != self.config.num_decoder_layers and kwargs.get("use_cache", True):
            kwargs.setdefault("past_key_values", EncoderDecoderCache(
                DynamicCache(config=self.model.decoder.config),
                DynamicCache(config=self.model.decoder.config),
            ))
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
