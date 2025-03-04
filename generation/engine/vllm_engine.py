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
    sampling_params = SamplingParams(max_tokens=max_new_tokens,
                                     stop=stop,
                                     temperature=0)
    vllm_completions = vllm.generate(prompts, sampling_params)

    result = []
    for vllm_completion in vllm_completions:
        result.append(CompletionObject(completion=vllm_completion.outputs[0].text,
                        prompt=vllm_completion.prompt,
                        num_prompt_tokens=len(vllm_completion.prompt_token_ids),
                        num_completion_tokens=len(vllm_completion.outputs[0].token_ids)))
    return result