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

def retrieve_top_k(query_text, bm25_obj, k=10):
    """
    Given a query_text and a BM25 object, returns a list of (document index, score)
    for the top k documents (only those with a nonzero score).
    """
    query_tokens = simple_tokenize(query_text)
    scores = bm25_obj.get_scores(query_tokens)
    idx_score_pairs = list(enumerate(scores))
    sorted_pairs = sorted(idx_score_pairs, key=operator.itemgetter(1), reverse=True)
    top_k = [(idx, score) for idx, score in sorted_pairs if score > 0][:k]
    return top_k

def augment_prompts_with_context_for_repo(code_chunks_path, prompts, output_path, top_k=10):
    """
    For a given repository, loads its code-chunks file and the corresponding unfinished prompts,
    then uses BM25 to retrieve the top matching code chunks.
    Augments each prompt with:
      - "context": a list of code chunk ids
      - "bm25_scores": the corresponding BM25 scores
    """
    code_chunks = load_jsonl(code_chunks_path)
    corpus_tokens, id_map = build_bm25_index(code_chunks)
    bm25 = BM25(corpus_tokens)
    new_prompts = []
    for prompt_obj in prompts:
        query_text = prompt_obj.get("prompt", "")
        top_matches = retrieve_top_k(query_text, bm25, k=top_k)
        # Sort descending by BM25 score.
        top_matches = sorted(top_matches, key=lambda x: x[1], reverse=True)
        context_ids = [id_map[idx] for idx, score in top_matches]
        scores = [score for idx, score in top_matches]
        prompt_obj["context"] = context_ids
        prompt_obj["bm25_scores"] = scores
        new_prompts.append(prompt_obj)
    dump_jsonl(new_prompts, output_path)
    print(f"Augmented prompts written to {output_path}")

def augment_all_repos_prompts(repo_names, data_schema_dir, repo_eval_dir, test_types, output_dir, top_k=10):
    """
    For each repository and for each test type, filters unfinished prompts from the corresponding test file
    and augments them using BM25 with the repository's code-chunks file.
    The output filenames include the test type.
    """
    for test_type in test_types:
        test_filepath = os.path.join(repo_eval_dir, f"{test_type}.test.jsonl")
        if not os.path.exists(test_filepath):
            print(f"Test file {test_filepath} not found. Skipping test type {test_type}.")
            continue
        print(f"Processing test type: {test_type}")
        # Load prompts from the test file for this test type.
        all_prompts = load_jsonl(test_filepath)
        for i, repo in enumerate(repo_names):
            print(f"Processing repo {i+1}/{len(repo_names)}: {repo}")
            # Filter prompts that belong to this repository (assuming task_id is formatted as "repo/...")
            repo_prompts = [p for p in all_prompts if p.get("metadata", {}).get("task_id", "").split('/')[0] == repo]
            if not repo_prompts:
                print(f"No prompts found for repo {repo} using test type {test_type}. Skipping BM25 augmentation for this repo.")
                continue
            code_chunks_path = os.path.join(data_schema_dir, f"code-chunks_{repo}.jsonl")
            output_path = os.path.join(output_dir, f"unfinished-code-w-context_{repo}_{test_type}_bm25.jsonl")
            augment_prompts_with_context_for_repo(code_chunks_path, repo_prompts, output_path, top_k)

def main():
    parser = argparse.ArgumentParser(description="Augment unfinished prompts with BM25 context per repository for multiple test types.")
    parser.add_argument("--repo_eval_dir", type=str, default="RepoEval-Updated",
                        help="Directory containing the RepoEval test files.")
    parser.add_argument("--test_types", type=str, default="api_level.java,api_level.python,line_level.java,line_level.python",
                        help="Comma-separated list of test types to process.")
    args = parser.parse_args()

    test_types = [t.strip() for t in args.test_types.split(',') if t.strip()]

    repo_names = [
        "Aelysium-Group_rusty-connector",
        "apple_axlearn",
        "awslabs_fortuna",
        "devchat",
        "FloatingPoint-MC_MIN",
        "gentics_cms-oss",
        "Guiqu1aixi_rocketmq",
        "huggingface_diffusers",
        "itlemon_chatgpt4j",
        "metagpt",
        "mybatis-flex_mybatis-flex",
        "nemo_aligner",
        "neoforged_NeoGradle",
        "nerfstudio-project_nerfstudio",
        "Open-DBT_open-dbt",
        "opendilab_ACE",
        "QingruZhang_AdaLoRA",
        "QuasiStellar_custom-pixel-dungeon",
        "SimonHalvdansson_Harmonic-HN",
        "task_weaver"
    ]
    data_schema_dir = "code-chunks"        # Where code-chunks files (from process_repos.py) are stored.
    output_prompts_dir = "augmented-prompts" # Directory where BM25-augmented prompt files will be saved.
    os.makedirs(output_prompts_dir, exist_ok=True)
    repo_eval_dir = args.repo_eval_dir

    augment_all_repos_prompts(repo_names, data_schema_dir, repo_eval_dir, test_types, output_prompts_dir, top_k=10)

if __name__ == '__main__':
    main()
