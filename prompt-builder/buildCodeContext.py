import os
import json
import math
import operator
from collections import defaultdict

# --- Helper functions ---

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

# --- BM25 Implementation ---

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
        self.df = defaultdict(int)  # document frequency for each term
        self.idf = {}             # inverse document frequency for each term
        self._initialize()

    def _initialize(self):
        for doc in self.corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    self.df[term] += 1
                    seen.add(term)
        # Compute idf per term
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
    Build BM25 index for a list of code chunks.
    Each chunk is a dict with a "code" field.
    Returns:
      corpus_tokens: list of token lists for each code chunk.
      id_map: list mapping BM25 index to the code chunk's id.
    """
    corpus_tokens = []
    id_map = []
    for chunk in chunks:
        text = chunk.get("code", "")
        tokens = simple_tokenize(text)
        corpus_tokens.append(tokens)
        id_map.append(chunk["id"])
    return corpus_tokens, id_map

def retrieve_top_k(query_text, bm25_obj, k=10):
    """
    Given a query_text and a BM25 object, returns a list of (document index, score)
    for the top k documents (only those with a nonzero score).
    """
    query_tokens = simple_tokenize(query_text)
    scores = bm25_obj.get_scores(query_tokens)
    idx_score_pairs = list(enumerate(scores))
    # Sort by descending score
    sorted_pairs = sorted(idx_score_pairs, key=operator.itemgetter(1), reverse=True)
    top_k = [(idx, score) for idx, score in sorted_pairs if score > 0][:k]
    return top_k

# --- BM25 Retrieval for Prompt Building (repository-aware) ---

def augment_prompts_with_context(code_chunks_path, unfinished_prompts_path, output_path, top_k=10):
    """
    Reads the code-chunks corpus and the unfinished prompt file,
    then for each prompt (from a specific repository) uses BM25 to find top k matching code chunks
    from the same repository.
    
    The output JSONL file will include, for each prompt, two additional fields:
      - "context": a list of code chunk ids
      - "bm25_scores": a list of the corresponding BM25 scores
    """
    # Load code chunks and unfinished prompts.
    code_chunks = load_jsonl(code_chunks_path)
    prompts = load_jsonl(unfinished_prompts_path)
    
    # Group code chunks by repository.
    repo_to_chunks = defaultdict(list)
    for chunk in code_chunks:
        repo = chunk["metadata"].get("repository")
        if not repo and "fpath_tuple" in chunk["metadata"]:
            repo = chunk["metadata"]["fpath_tuple"][0]
        if repo is not None:
            repo_to_chunks[repo].append(chunk)
    
    # Build BM25 index per repository.
    repo_to_bm25 = {}
    repo_to_id_map = {}
    for repo, chunks in repo_to_chunks.items():
        corpus_tokens, id_map = build_bm25_index(chunks)
        bm25 = BM25(corpus_tokens)
        repo_to_bm25[repo] = bm25
        repo_to_id_map[repo] = id_map
    
    # For each prompt, determine its repository and perform BM25 retrieval.
    new_prompts = []
    for prompt_obj in prompts:
        meta = prompt_obj.get("metadata", {})
        # Prefer the repository from task_id if available.
        repo = None
        if "task_id" in meta:
            repo = meta["task_id"].split('/')[0]
        elif "fpath_tuple" in meta and meta["fpath_tuple"]:
            repo = meta["fpath_tuple"][0]
        
        if repo is None or repo not in repo_to_bm25:
            prompt_obj["context"] = []
            prompt_obj["bm25_scores"] = []
        else:
            bm25 = repo_to_bm25[repo]
            id_map = repo_to_id_map[repo]
            query_text = prompt_obj.get("prompt", "")
            top_matches = retrieve_top_k(query_text, bm25, k=top_k)
            # Sort descending by score.
            top_matches = sorted(top_matches, key=lambda x: x[1], reverse=True)
            context_ids = [id_map[idx] for idx, score in top_matches]
            scores = [score for idx, score in top_matches]
            prompt_obj["context"] = context_ids
            prompt_obj["bm25_scores"] = scores
        new_prompts.append(prompt_obj)
    
    dump_jsonl(new_prompts, output_path)
    print(f"Augmented prompts with BM25 context written to {output_path}")

# --- Main ---

def main():
    # File paths (adjust as needed)
    code_chunks_path = "/u/dylandunham/CS598LMZ/prompt-builder/data-schema/code-chunks_Aelysium-Group_rusty-connector.jsonl"
    unfinished_prompts_path = "/u/dylandunham/CS598LMZ/prompt-builder/RepoEval-Updated/api_level.java.test.jsonl"
    output_path = "/u/dylandunham/CS598LMZ/prompt-builder/data-schema/unfinished-code-w-context_bm25.jsonl"
    
    # Retrieve top 10 context chunks using BM25 per repository.
    augment_prompts_with_context(code_chunks_path, unfinished_prompts_path, output_path, top_k=10)

if __name__ == '__main__':
    main()
