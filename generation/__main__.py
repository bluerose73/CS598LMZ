# python -m generation generation-config.yaml

import argparse
import yaml
import json
from .format import batch_format_prompt
from .engine import vllm_generate

parser = argparse.ArgumentParser(description="Run LLM generation")
parser.add_argument("config", type=str, help="Path to the config YAML file")

args = parser.parse_args()
config_path = args.config

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

code_chunks_list = []
unfinished_code_with_context_list = []
with open(config["code-chunks-path"], "r") as f:
    for line in f:
        code_chunks_list.append(json.loads(line))
with open(config["unfinished-code-with-context-path"], "r") as f:
    for line in f:
        unfinished_code_with_context_list.append(json.loads(line))



if config["architecture"] == "decoder":
    prompt_list = batch_format_prompt(code_chunks_list,
                                      unfinished_code_with_context_list,
                                      config["prompt-format"])
    
    if config["engine"] == "vllm":

        if config["task"] == "line-completion":
            print("Running line completion. Setting stop token to newline.")
            stop = ["\n"]
        else:
            stop = None

        completions = vllm_generate(config["model"],
                                    prompt_list,
                                    config["max-new-tokens"],
                                    stop)
    elif config["engine"] == "transformer":
        raise NotImplementedError("Transformer engine not implemented yet.")
    else:
        raise ValueError(f"Unknown engine: {config['engine']}")

elif config["architecture"] == "FiD":
    assert config["engine"] == "transformer", "VLLM doesn't support FiD"


with open(config["output-path"], "w") as f:
    for completion in completions:
        f.write(completion.model_dump_json() + "\n")