import argparse
from fid.trainer.dataset import FidTestDataset
from fid.generation.generation import qwen2_batch_generate, DecoderTestDataCollator
from fid.model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM
from transformers.models.qwen2 import Qwen2TokenizerFast
from torch.utils.data import DataLoader
import torch
import os
import json
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Qwen2 generation for RepoEval.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to the input tokenized data directory.")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for generated completions.")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", torch_dtype="auto", device_map="auto")

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")
dataset = FidTestDataset(
    tokenizer=tokenizer,
    tokenized_data_load_dir=args.input_dir,
)

# Base dataloader
base_dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=DecoderTestDataCollator(tokenizer.pad_token_id, cross_file_context=False),
    num_workers=4,
    shuffle=False,
    drop_last=False,
)

output_jsonl_path = os.path.join(output_dir, "qwen2-base-completion.jsonl")
with open(output_jsonl_path, "w", encoding="utf-8") as output_file:
    with torch.no_grad():
        for i, batch in enumerate(tqdm(base_dataloader)):
            completions: list[dict] = qwen2_batch_generate(
                decoder,
                tokenizer,
                batch,
                "cuda:0",
            )
            for completion in completions:
                output_file.write(json.dumps(completion) + "\n")

# RAG dataloader
rag_dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=DecoderTestDataCollator(tokenizer.pad_token_id, cross_file_context=True),
    num_workers=4,
    shuffle=False,
    drop_last=False,
)

output_jsonl_path = os.path.join(output_dir, "qwen2-rag-completion.jsonl")
with open(output_jsonl_path, "w", encoding="utf-8") as output_file:
    with torch.no_grad():
        for i, batch in enumerate(tqdm(rag_dataloader)):
            completions: list[dict] = qwen2_batch_generate(
                decoder,
                tokenizer,
                batch,
                "cuda:0",
            )
            for completion in completions:
                output_file.write(json.dumps(completion) + "\n")