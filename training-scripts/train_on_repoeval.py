from trainer.fid_datamodule import FidTrainingDataModule
from trainer.fid_trainer import FiDLightningModule
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
import torch

# TF32 tensor cores
torch.set_float32_matmul_precision("high")

encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
encoder.gradient_checkpointing_enable()
config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
lr = 0.0001

logger = WandbLogger(
    name="train-repoeval-updated",
    save_dir="./wandb-logs",
)

datamodule = FidTrainingDataModule(
    tokenized_data_load_dir="./data/repoeval-updated-pathdist/tokenized",
    tokenizer=tokenizer,
    batch_size=4,
)


fidmodule = FiDLightningModule(encoder, decoder, lr)

trainer = L.Trainer(deterministic=True,
                    precision="bf16-mixed",
                    accumulate_grad_batches=16,
                    max_epochs=5,
                    logger=logger,
)

trainer.fit(fidmodule, datamodule)
