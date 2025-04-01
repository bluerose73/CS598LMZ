import argparse
import random
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Qwen2TokenizerFast
import numpy as np

def get_all_lines_from_repos(repo_paths):
    lines = []
    for repo_path in tqdm(repo_paths, desc="Reading repos"):
        for root, _, files in os.walk(repo_path):
            for fname in files:
                try:
                    with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                        lines.extend(f.readlines())
                except:
                    continue  # skip unreadable files
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--num_repos', required=True, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    args = parser.parse_args()

    random.seed(args.random_seed)
    input_dir = Path(args.input_dir)

    all_repos = [entry for entry in input_dir.iterdir() if entry.is_dir()]
    sampled_repos = random.sample(all_repos, min(args.num_repos, len(all_repos)))

    lines = get_all_lines_from_repos(sampled_repos)
    print(f"Total lines collected: {len(lines)}")

    tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")
    token_counts = []
    for line in tqdm(lines, desc="Tokenizing lines"):
        line = line.strip()
        if line:  # skip empty lines
            tokens = tokenizer.tokenize(line)
            token_counts.append(len(tokens))

    token_counts = np.array(token_counts)
    mean_tokens = np.mean(token_counts)
    p99_tokens = np.percentile(token_counts, 99)

    print(f"Mean tokens per line: {mean_tokens:.2f}")
    print(f"99% of lines have <= {p99_tokens:.0f} tokens")

    # Plotting CDF
    sorted_counts = np.sort(token_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_counts, cdf, label='CDF')
    plt.axhline(y=0.99, color='gray', linestyle='--', label='CDF 99%')

    plt.xscale('log')
    plt.xlabel('Number of tokens in a line (log scale)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Token Count per Line')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("token_distribution.png", dpi=300)

if __name__ == "__main__":
    main()
