from trainer.fid_datamodule import FidTrainingDataset
from transformers.models.qwen2 import Qwen2TokenizerFast

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")

dataset = FidTrainingDataset(
    code_chunks_dir=r"./data/repoeval-updated-pathdist/code-chunks",
    code_to_complete_dir=r"./data/repoeval-updated-pathdist/code-to-complete",

    tokenizer=tokenizer,
    tokenized_data_save_dir="./data/repoeval-updated-pathdist/tokenized",
)