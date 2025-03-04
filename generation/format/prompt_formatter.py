def format_prompt_repocoder_python(unfinished_code: dict, context: list[dict]) -> str:
    """
    RepoCoder's prompt format.
    Put context in a comment, and concatenate it with the unfinished code.
    """
    return ""


def format_prompt_qwen_repo_level_completion(unfinished_code: dict, context: list[dict]) -> str:
    """
    Format a prompt using the repo-level completion format by Qwen-2.5-coder
    """
    repo_name_line = f"<|repo_name|>{unfinished_code['metadata']['fpath_tuple'][0]}"
    
    context_lines = []
    for chunk in context:
        file_path = "/".join(chunk["metadata"]["fpath_tuple"][1:])
        file_content = chunk["code"]
        line = f"<|file_sep|>{file_path}\n{file_content}"
        context_lines.append(line)

    prompt_str = "\n".join([repo_name_line] + context_lines)
    return prompt_str