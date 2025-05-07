from typing import List
import re
from .types import CodeChunk, CodeToComplete
from rank_bm25 import BM25Okapi

# -------------------------------------------------------------------
# Retriever base class and a path distance implementation

class Retriever:
    def retrieve(self, target_chunk: CodeChunk, code_chunks: List[CodeChunk], n_context: int, cross_file: bool = False) -> List[int]:
        raise NotImplementedError("This method should be overridden by subclasses.")


class PathDistanceRetriever(Retriever):
    def retrieve(self, target_chunk: CodeChunk, code_chunks: List[CodeChunk], n_context: int,
                 cross_file: bool = False) -> List[int]:
        def path_distance(tuple1: List[str], tuple2: List[str]) -> int:
            # Compute distance based on common prefix length.
            common = 0
            for a, b in zip(tuple1, tuple2):
                if a == b:
                    common += 1
                else:
                    break
            return (len(tuple1) - common) + (len(tuple2) - common)
        
        distances = []
        for chunk in code_chunks:
            if chunk.id == target_chunk.id:
                continue  # skip the target itself
            if cross_file and chunk.fpath_tuple == target_chunk.fpath_tuple:
                continue
            dist = path_distance(target_chunk.fpath_tuple, chunk.fpath_tuple)
            distances.append((dist, chunk.id))
        # Sort by distance: smaller distance means more similar path
        distances.sort(key=lambda x: x[0])
        # Return the top n_context code chunk ids
        return [chunk_id for _, chunk_id in distances[:n_context]]


class BM25Retriever(Retriever):
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self.index_ready = False

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def setup(self, code_chunks: List[CodeChunk], cross_file: bool = False, target_chunk: CodeChunk = None):
        corpus = []
        chunk_ids = []

        for chunk in code_chunks:
            if target_chunk and chunk.id == target_chunk.id:
                continue
            if cross_file and target_chunk and chunk.fpath_tuple == target_chunk.fpath_tuple:
                continue
            tokens = self.tokenize(chunk.code)
            corpus.append(tokens)
            chunk_ids.append(chunk.id)

        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = chunk_ids
        self.index_ready = True

    def retrieve(self, target_chunk: CodeChunk, code_chunks: List[CodeChunk], n_context: int,
                 cross_file: bool = False) -> List[int]:
        if not self.index_ready:
            self.setup(code_chunks, cross_file=cross_file, target_chunk=target_chunk)

        query_tokens = self.tokenize(target_chunk.code)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_context]
        return [self.chunk_ids[i] for i in top_indices]