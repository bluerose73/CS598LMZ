import os
import json
import math
import operator
from collections import defaultdict
import argparse

def load_jsonl(filepath):
    """Loads a JSON Lines file and returns a list of JSON objects."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def dump_jsonl(data, filepath):
    """Writes a list of JSON objects to a JSON Lines file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def simple_tokenize(text):
    """A simple whitespace tokenizer that lowercases the text."""
    return text.lower().split()

class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        """
        corpus_tokens: list of lists; each inner list is the tokenized document.
        k1, b: BM25 parameters.
        """
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / self.N if self.N > 0 else 0
        self.k1 = k1
        self.b = b
        self.df = defaultdict(int)
        self.idf = {}
        self._initialize()

    def _initialize(self):
        for doc in self.corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    self.df[term] += 1
                    seen.add(term)
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query_tokens):
        """Computes BM25 scores for all documents in the corpus given query_tokens."""
        scores = [0.0] * self.N
        for idx, doc in enumerate(self.corpus):
            doc_len = len(doc)
            score = 0.0
            for term in query_tokens:
                if term not in self.idf:
                    continue
                freq = doc.count(term)
                if freq == 0:
                    continue
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += numerator / denominator
            scores[idx] = score
        return scores

def build_bm25_index(chunks):
    """
    Builds a BM25 index from a list of code chunks.
    Each chunk is expected to have a "code" field.
    Returns:
      - corpus_tokens: a list of token lists for each code chunk.
      - id_map: a list mapping BM25 index positions to the code chunk's id.
    """
    corpus_tokens = []
    id_map = []
    for chunk in chunks:
        text = chunk.get("code", "")
        tokens = simple_tokenize(text)
        corpus_tokens.append(tokens)
        id_map.append(chunk["id"])
    return corpus_tokens, id_map

def retrieve_top_k(query_text, bm25_obj, code_chunks, id_map, k=10, exclude_current_file=True, current_file_path=None):
    """
    Given a query_text and a BM25 object, returns a list of code chunk ids and scores
    for the top k chunks (only those with a nonzero score).
    
    Args:
        query_text: The text to search for
        bm25_obj: BM25 object containing the index
        code_chunks: List of code chunks to reference metadata
        id_map: Mapping from index positions to chunk ids
        k: Number of top results to return
        exclude_current_file: Whether to exclude chunks from the current file
        current_file_path: The current file's path (needed if exclude_current_file is True)
    """
    query_tokens = simple_tokenize(query_text)
    scores = bm25_obj.get_scores(query_tokens)
    idx_score_pairs = list(enumerate(scores))
    
    # Filter out current file if needed
    if exclude_current_file and current_file_path:
        filtered_pairs = []
        for idx, score in idx_score_pairs:
            chunk_fpath = "/".join(code_chunks[idx].get("metadata", {}).get("fpath_tuple", [])[1:])
            if chunk_fpath != current_file_path:
                filtered_pairs.append((idx, score))
        idx_score_pairs = filtered_pairs

    sorted_pairs = sorted(idx_score_pairs, key=operator.itemgetter(1), reverse=True)
    # Filter out zero scores and take top k
    top_k_pairs = [(idx, score) for idx, score in sorted_pairs if score > 0][:k]
    
    # Map to chunk ids and scores
    result_ids = [id_map[idx] for idx, _ in top_k_pairs]
    result_scores = [score for _, score in top_k_pairs]
    
    return result_ids, result_scores

def augment_repo_with_context(repo_name, train_dir, chunks_dir, output_dir, top_k=10):
    """
    Augments training data for a repository with context from code chunks.
    
    Args:
        repo_name: Name of the repository
        train_dir: Directory containing full file training data
        chunks_dir: Directory containing code chunks
        output_dir: Directory to write augmented training data
        top_k: Number of top context chunks to include
    """
    # Load training data and code chunks
    train_path = os.path.join(train_dir, f"{repo_name}.jsonl")
    chunks_path = os.path.join(chunks_dir, f"{repo_name}.jsonl")
    
    if not os.path.exists(train_path):
        print(f"Warning: Training file for {repo_name} not found at {train_path}")
        return
    
    if not os.path.exists(chunks_path):
        print(f"Warning: Chunks file for {repo_name} not found at {chunks_path}")
        return
    
    train_data = load_jsonl(train_path)
    code_chunks = load_jsonl(chunks_path)
    
    # Build BM25 index from code chunks
    corpus_tokens, id_map = build_bm25_index(code_chunks)
    bm25 = BM25(corpus_tokens)
    
    # Augment each training entry with context
    augmented_data = []
    for entry in train_data:
        query_text = entry.get("prompt", "")
        file_path = entry.get("file_path", "")
        
        # Retrieve top-k chunks and their scores
        context_ids, scores = retrieve_top_k(
            query_text, 
            bm25, 
            code_chunks,
            id_map,
            k=top_k,
            exclude_current_file=True,
            current_file_path=file_path
        )
        
        # Add context and scores to the entry
        entry["context"] = context_ids
        entry["bm25_scores"] = scores
        augmented_data.append(entry)
    
    # Write augmented data
    output_path = os.path.join(output_dir, f"{repo_name}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump_jsonl(augmented_data, output_path)
    print(f"Augmented data for {repo_name} written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Augment training data with BM25 context from code chunks.")
    parser.add_argument("--train-dir", type=str, default="train",
                        help="Directory containing full file training data")
    parser.add_argument("--chunks-dir", type=str, default="training-code-chunks",
                        help="Directory containing code chunks")
    parser.add_argument("--output-dir", type=str, default="train-with-context",
                        help="Directory to write augmented training data")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top context chunks to include")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of repositories from train directory
    repos = []
    for filename in os.listdir(args.train_dir):
        if filename.endswith(".jsonl"):
            repo_name = filename[:-6]  # Remove .jsonl extension
            repos.append(repo_name)
    
    print(f"Found {len(repos)} repositories in {args.train_dir}")
    
    # Process each repository
    for i, repo in enumerate(repos):
        print(f"Processing repository {i+1}/{len(repos)}: {repo}")
        augment_repo_with_context(
            repo,
            args.train_dir,
            args.chunks_dir,
            args.output_dir,
            top_k=args.top_k
        )

if __name__ == "__main__":
    main()