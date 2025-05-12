from fid.trainer.fid_datamodule import FidTrainingDataModule
from fid.trainer.fid_trainer import FiDLightningModule
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from fid.model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
import torch

L.seed_everything(42)

# TF32 tensor cores
torch.set_float32_matmul_precision("high")

encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype="auto")
encoder.gradient_checkpointing_enable()
# encoder = torch.compile(encoder, fullgraph=True, dynamic=True)

config = Qwen2FidDecoderConfig.from_json_file("./fid/model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, torch_dtype="auto")
# decoder = torch.compile(decoder, fullgraph=True, dynamic=True)


tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
lr = 0.0003


# layers[35].cross_attn_layernorm.weight
# print(f"layers[35].cross_attn_layernorm.weight: {decoder.model.layers[35].cross_attn_layernorm.weight}")


effective_batch_size = 128
n_devices = torch.cuda.device_count()
per_device_batch_size = 8
gradient_accumulation_steps = effective_batch_size // (n_devices * per_device_batch_size)
print(f"effective_batch_size: {effective_batch_size}, n_devices: {n_devices}, per_device_batch_size: {per_device_batch_size}, gradient_accumulation_steps: {gradient_accumulation_steps}")

logger = WandbLogger(
    name="train-the-stack-v2-20k",
    save_dir="./wandb-logs",
)
lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")

datamodule = FidTrainingDataModule(
    tokenized_data_load_dir="/work/nvme/becw/sma2/the-stack-v2-20k/tokenized",
    tokenizer=tokenizer,
    batch_size=per_device_batch_size,
    copy_ratio=0.5,
    # post_process_max_context_num=1,
)


fidmodule = FiDLightningModule(encoder, decoder, pad_token_id=tokenizer.pad_token_id, lr=lr)

trainer = L.Trainer(deterministic=True,
                    precision="bf16-mixed",
                    accumulate_grad_batches=gradient_accumulation_steps,
                    max_epochs=3,
                    logger=logger,
                    log_every_n_steps=10,
                    val_check_interval=0.5,
                    # detect_anomaly=True,
                    callbacks=[lr_monitor],
)

trainer.fit(fidmodule, datamodule)
