#!/usr/bin/env python3
import os
import json
import argparse
import voyageai
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Count real Voyage-AI tokens for each code chunk repo")
    parser.add_argument(
        "--chunks-dir",
        default="code-chunks-smart",
        help="Directory containing code-chunks_<repo>.jsonl files")
    parser.add_argument(
        "--model",
        default="voyage-code-3",
        help="Voyage model to use for tokenization/counting")
    args = parser.parse_args()

    # initialize client (will pick up VOYAGE_API_KEY from env)
    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    total_tokens_all = 0
    total_chunks_all = 0

    for fn in sorted(os.listdir(args.chunks_dir)):
        if not fn.startswith("code-chunks_") or not fn.endswith(".jsonl"):
            continue

        repo_name = fn[len("code-chunks_"):-len(".jsonl")]
        path = os.path.join(args.chunks_dir, fn)

        # collect all the code snippets
        snippets = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                snippets.append(obj["code"])

        n_chunks = len(snippets)
        if n_chunks == 0:
            print(f"{repo_name}: no chunks, skipping.")
            continue

        # count tokens for the whole list at once
        repo_tokens = client.count_tokens(snippets, model=args.model)

        print(f"{repo_name}: {n_chunks} chunks, {repo_tokens} tokens (~{repo_tokens/1000:.2f}K)")

        total_chunks_all += n_chunks
        total_tokens_all += repo_tokens

    print("\n— SUMMARY —")
    print(f"Total chunks : {total_chunks_all}")
    print(f"Total tokens : {total_tokens_all}  (~{total_tokens_all/1000:.2f}K)")

if __name__ == "__main__":
    main()

# (.venv) dyland@Dylans-MacBook-Air prompt-builder % python count_chunk_tokens.py
# tokenizer.json: 100%|██████████████████████████████████████████████| 7.03M/7.03M [00:00<00:00, 17.0MB/s]
# Aelysium-Group_rusty-connector: 2297 chunks, 269821 tokens (~269.82K)
# FloatingPoint-MC_MIN: 52199 chunks, 13335654 tokens (~13335.65K)
# Guiqu1aixi_rocketmq: 9606 chunks, 1679987 tokens (~1679.99K)
# Open-DBT_open-dbt: 2116 chunks, 339051 tokens (~339.05K)
# QingruZhang_AdaLoRA: 30350 chunks, 9243347 tokens (~9243.35K)
# QuasiStellar_custom-pixel-dungeon: 10135 chunks, 2123258 tokens (~2123.26K)
# SimonHalvdansson_Harmonic-HN: 765 chunks, 174985 tokens (~174.99K)
# apple_axlearn: 6869 chunks, 2154670 tokens (~2154.67K)
# awslabs_fortuna: 962 chunks, 255547 tokens (~255.55K)
# devchat: 234 chunks, 39879 tokens (~39.88K)
# gentics_cms-oss: 29143 chunks, 7951066 tokens (~7951.07K)
# huggingface_diffusers: 3657 chunks, 1473967 tokens (~1473.97K)
# itlemon_chatgpt4j: 719 chunks, 62315 tokens (~62.31K)
# metagpt: 1826 chunks, 281947 tokens (~281.95K)
# mybatis-flex_mybatis-flex: 5930 chunks, 750982 tokens (~750.98K)
# nemo_aligner: 379 chunks, 97268 tokens (~97.27K)
# neoforged_NeoGradle: 2010 chunks, 316625 tokens (~316.62K)
# nerfstudio-project_nerfstudio: 1073 chunks, 308745 tokens (~308.75K)
# opendilab_ACE: 3581 chunks, 965788 tokens (~965.79K)
# task_weaver: 750 chunks, 150369 tokens (~150.37K)

# — SUMMARY —
# Total chunks : 164601
# Total tokens : 41975271  (~41975.27K)