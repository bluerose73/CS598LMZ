from pydantic import BaseModel

class CodeChunk(BaseModel):
    code: str
    id: int
    repository: str
    fpath_tuple: list[str]
    metadata: dict = {}


class CodeToComplete(BaseModel):
    code: str
    context: list[int]  # List of code chunk IDs
    repository: str
    fpath_tuple: list[str]
    metadata: dict = {}

