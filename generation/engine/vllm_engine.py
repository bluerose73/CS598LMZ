from vllm import LLM, SamplingParams
from ..types import CompletionObject

def vllm_generate(model: str,
                  prompts: list[str],
                  max_new_tokens: int = 128,
                  stop: list[str] | None = None) -> list[CompletionObject]:
    """
    Generate completions for a list of prompts using the VLLM model.
    """
    vllm = LLM(model)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, stop=stop)
    completions = vllm.generate(prompts, sampling_params)

    result = []
    for completion in completions:
        result.append(CompletionObject(completion=completion.outputs[0].text,
                        prompt=completion.prompt,
                        num_prompt_tokens=len(completion.prompt_token_ids),
                        num_completion_tokens=len(completion.outputs[0].token_ids)))
    return completions