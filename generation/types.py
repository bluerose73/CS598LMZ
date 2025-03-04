from pydantic import BaseModel

class CompletionObject(BaseModel):
    completion: str
    prompt: str
    num_prompt_tokens: int
    num_completion_tokens: int
    latency: float | None = None