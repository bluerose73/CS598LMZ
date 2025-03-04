from .prompt_formatter import *
from tqdm import tqdm
from typing import Callable

def get_formatter(format: str) -> Callable[[dict, list[dict]], str]:
    """
    Get the formatter function by name.
    """
    if format == "repocoder-python":
        return format_prompt_repocoder_python
    elif format == "qwen-repo-level-completion":
        return format_prompt_qwen_repo_level_completion
    else:
        raise ValueError(f"Unknown format: {format}")



def batch_format_prompt(chunk_list: list[dict], unfinished_code_w_context_list: list[dict], format: str) -> list[str]:
    """
    Format a list of prompts.
    """
    formatter = get_formatter(format)

    prompt_strings = []
    for unfinished_code_w_context in tqdm(unfinished_code_w_context_list, desc="Formatting prompts"):
        context = [chunk_list[i] for i in unfinished_code_w_context["context"]]
        prompt_strings.append(formatter(unfinished_code_w_context, context))