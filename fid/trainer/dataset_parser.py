from ..data_util.types import CodeChunk, CodeToComplete
import json

def parse_code_chunk(line: str) -> CodeChunk:
    data = json.loads(line)
    repository = data['metadata']['repository']
    code_chunk = CodeChunk(
        code=data['code'],
        id = data['id'],
        repository=repository,
        fpath_tuple=data['metadata']['fpath_tuple'][1:],
        metadata=data['metadata']
    )
    del code_chunk.metadata['fpath_tuple']
    del code_chunk.metadata['repository']
    return code_chunk


def parse_unfinished_code(line: str) -> CodeToComplete:
    data = json.loads(line)
    repository = data['metadata']['task_id'].split('/')[0]
    code_to_complete = CodeToComplete(
        code=data['prompt'],
        context=data['context'],
        repository=repository,
        fpath_tuple=data['metadata']['fpath_tuple'][1:],
        metadata=data['metadata']
    )
    del code_to_complete.metadata['fpath_tuple']
    return code_to_complete