from trainer.fid_datamodule import FidTrainingDataset
from transformers.models.qwen2 import Qwen2TokenizerFast

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")

dataset = FidTrainingDataset(
    code_chunks_dir=r"/work/nvme/becw/sma2/the-stack-v2-20k/code-chunks",
    code_to_complete_dir=r"/work/nvme/becw/sma2/the-stack-v2-20k/code-to-complete",

    tokenizer=tokenizer,
    tokenized_data_save_dir="/work/nvme/becw/sma2/the-stack-v2-20k/tokenized",
)