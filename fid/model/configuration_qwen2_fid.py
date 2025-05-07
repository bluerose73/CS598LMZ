from transformers import PretrainedConfig  # Import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config


# class Qwen2FidEncoderConfig(Qwen2Config):
#     pass


class Qwen2FidDecoderConfig(Qwen2Config):
    model_type = "qwen2-fid-decoder"

    def __init__(self,
                 num_cross_attn_layers=1,
                 encoder_hidden_size=896,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_cross_attn_layers = num_cross_attn_layers
        self.encoder_hidden_size = encoder_hidden_size


# class Qwen2FidConfig(PretrainedConfig):
#     model_type = "qwen2-fid"
#     is_composition = True

#     def __init__(self, decoder_config, encoder_config, **kwargs):
#         super().__init__(**kwargs)
#         self.decoder_config = decoder_config
#         self.encoder_config = encoder_config

