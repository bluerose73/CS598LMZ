import torch
import os
import sys
sys.path.append('/work/nvme/becw/sma2/cs598lmz')

from fid.trainer.dataset import FidTestDataset
from fid.trainer.fid_trainer import FiDLightningModule
from fid.model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM, Qwen2FidDecoderConfig
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from torch.utils.data import DataLoader


encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype="auto", device_map="auto")

config = Qwen2FidDecoderConfig.from_json_file("/work/nvme/becw/sma2/cs598lmz/fid/model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, torch_dtype="auto", device_map="auto")
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

encoder_old_weights = encoder.layers[10].self_attn.q_proj.weight.detach().clone()
print(encoder_old_weights.size())

print(encoder_old_weights[0, :10])

model_path = r"/work/hdd/becw/sma2/cs598lmz/wandb-logs/lightning_logs/ahkjym1q/checkpoints/epoch=2-step=4335.ckpt"
# model_path = r"/work/nvme/becw/sma2/cs598lmz/wandb-logs/lightning_logs/1vz75kyi/checkpoints/epoch=0-step=1445.ckpt"

module = FiDLightningModule.load_from_checkpoint(model_path,
    encoder=encoder,
    decoder=decoder,
)

encoder = module.encoder

encoder_new_weights = encoder.layers[10].self_attn.q_proj.weight.detach().clone()
print(encoder_new_weights.size())

print(encoder_new_weights[0, :10])

print(encoder_old_weights == encoder_new_weights)