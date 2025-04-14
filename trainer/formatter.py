from .types import CodeChunk, CodeToComplete

class BaseCodeChunkFormatter:
    def __call__(self, code_chunk: CodeChunk) -> str:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseCodeToCompleteFormatter:
    def __call__(self, code_to_complete: CodeToComplete) -> str:
        raise NotImplementedError("Subclasses should implement this method.")


class PythonCommentCodeChunkFormatter(BaseCodeChunkFormatter):
    """
    Put file path in a python comment.
    """
    def __call__(self, code_chunk: CodeChunk) -> str:
        # Use the file path tuple to create a comment with the file path
        if code_chunk.fpath_tuple:
            fpath_tuple = code_chunk.fpath_tuple
            if code_chunk.repository:
                fpath_tuple = [code_chunk.repository] + fpath_tuple
            fpath = '/'.join(fpath_tuple)
            fpath_line = f"# {fpath}\n"
        else:
            fpath_line = ""
        
        # Add the code chunk itself
        code_chunk_str = fpath_line + code_chunk.code
        return code_chunk_str


class PythonCommentCodeToCompleteFormatter(BaseCodeToCompleteFormatter):
    """
    Put file path in a python comment.
    """
    def __call__(self, code_to_complete: CodeToComplete) -> str:
        # Use the file path tuple to create a comment with the file path
        if code_to_complete.fpath_tuple:
            fpath_tuple = code_to_complete.fpath_tuple
            if code_to_complete.repository:
                fpath_tuple = [code_to_complete.repository] + fpath_tuple
            fpath = '/'.join(fpath_tuple)
            fpath_line = f"# {fpath}\n"
        else:
            fpath_line = ""
        
        # Add the code to complete itself
        code_to_complete_str = fpath_line + code_to_complete.code
        return code_to_complete_str