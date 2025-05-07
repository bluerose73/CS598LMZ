from fid.trainer.dataset import FidTestDataset
from transformers.models.qwen2 import Qwen2TokenizerFast

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")

dataset = FidTestDataset(
    code_chunks_dir=r"data/repoeval-updated/java/code-chunks",
    code_to_complete_dir=r"data/repoeval-updated/java/code-to-complete",

    tokenizer=tokenizer,
    tokenized_data_save_dir="data/repoeval-updated/java/tokenized",
)

print(f"len(dataset): {len(dataset)}")
print(f"dataset[0]: {dataset[0]}")