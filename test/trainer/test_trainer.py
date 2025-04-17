from trainer.fid_datamodule import FidTrainingDataModule
from trainer.fid_trainer import FiDLightningModule
import lightning as L
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
import torch

# TF32 tensor cores
torch.set_float32_matmul_precision("high")


encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
lr = 0.0001


datamodule = FidTrainingDataModule(
    tokenized_data_load_dir="./data/repoeval-updated-pathdist/tokenized",
    tokenizer=tokenizer,
    batch_size=2,
)


fidmodule = FiDLightningModule(encoder, decoder, lr)

trainer = L.Trainer(fast_dev_run=True)
trainer.fit(fidmodule, datamodule)