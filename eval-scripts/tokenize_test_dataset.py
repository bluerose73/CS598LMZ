import argparse
from fid.trainer.dataset import FidTestDataset
from fid.trainer.dataset_parser import parse_code_chunk, parse_unfinished_code
from transformers.models.qwen2 import Qwen2TokenizerFast

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Tokenize test dataset.")
parser.add_argument("--code_chunks_dir", required=True, help="Path to the code chunks directory.")
parser.add_argument("--code_to_complete_dir", required=True, help="Path to the code to complete directory.")
parser.add_argument("--tokenized_data_save_dir", required=True, help="Path to save the tokenized data.")
parser.add_argument("--requires_parser", action="store_true", default=False, help="Whether to require parser. Set to True if the dataset is from prompt-builder.")
parser.add_argument("--code_chunks_filename_prefix", default="code-chunks_", help="Prefix for code chunks filename.")
parser.add_argument("--code_to_complete_filename_prefix", default="code-to-complete_", help="Prefix for code to complete filename.")
args = parser.parse_args()


code_chunk_parser, code_to_complete_parser = None, None
if args.requires_parser:
    code_chunk_parser = parse_code_chunk
    code_to_complete_parser = parse_unfinished_code

dataset = FidTestDataset(
    code_chunks_dir=args.code_chunks_dir,
    code_to_complete_dir=args.code_to_complete_dir,
    tokenizer=tokenizer,
    tokenized_data_save_dir=args.tokenized_data_save_dir,
    code_chunks_filename_prefix=args.code_chunks_filename_prefix,
    code_to_complete_filename_prefix=args.code_to_complete_filename_prefix,
    code_chunk_parser=code_chunk_parser,
    code_to_complete_parser=code_to_complete_parser,
)

print(f"len(dataset): {len(dataset)}")
