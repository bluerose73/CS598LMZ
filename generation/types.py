from pydantic import BaseModel

class CompletionObject(BaseModel):
    completion: str
    prompt: str
    num_prompt_tokens: str
    num_completion_tokens: str
    latency: float | None = None