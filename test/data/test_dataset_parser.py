from fid.trainer.dataset_parser import parse_code_chunk, parse_unfinished_code
from rich import print

with open(r"/work/nvme/becw/sma2/cs598lmz/prompt-builder/code-chunks/code-chunks_Aelysium-Group_rusty-connector.jsonl", "r") as f:
    for line in f:
        code_chunk = parse_code_chunk(line)
        print(code_chunk)
        break

with open(r"/work/nvme/becw/sma2/cs598lmz/prompt-builder/augmented-prompts/unfinished-code-w-context_Aelysium-Group_rusty-connector_api_level.java_bm25.jsonl", "r") as f:
    for line in f:
        code_to_complete = parse_unfinished_code(line)
        print(code_to_complete)
        break