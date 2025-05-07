from fid.trainer.dataset import FidTestDataset
from fid.trainer.fid_trainer import FiDLightningModule
from fid.generation.generation import fid_batch_generate, FidTestDataCollator
from fid.model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM, Qwen2FidDecoderConfig
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from torch.utils.data import DataLoader
import torch
import os
import json
from tqdm import tqdm

output_dir = "./data/repoeval-updated/java/completion"
os.makedirs(output_dir, exist_ok=True)
output_jsonl_path = os.path.join(output_dir, "fid-copy-completion.jsonl")

model_path = r"/work/nvme/becw/sma2/cs598lmz/wandb-logs/lightning_logs/1vz75kyi/checkpoints/epoch=2-step=4335.ckpt"

encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype="auto")
encoder.gradient_checkpointing_enable()

config = Qwen2FidDecoderConfig.from_json_file("./fid/model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, torch_dtype="auto")


tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
dataset = FidTestDataset(
    tokenizer=tokenizer,
    tokenized_data_load_dir="data/repoeval-updated/java/tokenized",
    # post_process_max_context_num=2,
)
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=FidTestDataCollator(tokenizer.pad_token_id),
    num_workers=4,
    shuffle=False,
    drop_last=False,
)


module = FiDLightningModule.load_from_checkpoint(model_path,
    encoder=encoder,
    decoder=decoder,
)


decoder = module.decoder
decoder.eval()
encoder = module.encoder
encoder.eval()

output_file = open(output_jsonl_path, "w", encoding="utf-8")

with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):
        completions: list[dict] = fid_batch_generate(
            encoder,
            decoder,
            tokenizer,
            batch,
            "cuda:0",
        )
        for completion in completions:
            output_file.write(json.dumps(completion) + "\n")
