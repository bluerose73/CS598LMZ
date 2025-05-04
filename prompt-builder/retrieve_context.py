#!/usr/bin/env python3
"""
For each test prompt, embed it and retrieve top-K code-chunk IDs from Chroma,
excluding any chunks from the same source file as the test.
"""
import os
import json
import argparse

import voyageai
import chromadb
from dotenv import load_dotenv

load_dotenv()

def main():
    p = argparse.ArgumentParser(
        description="Embed test prompts and retrieve top-K code chunks (minus same-file hits)"
    )
    p.add_argument("--test-file",
                   default="RepoEval-Updated/api_level.python.test.jsonl",
                   help="Input JSONL of tests (must have prompt + metadata.task_id + metadata.fpath_tuple)")
    p.add_argument("--out-file",
                   default="RepoEval-Updated-Context/api_level.python.test.jsonl",
                   help="Where to write augmented JSONL with embeddings & context")
    p.add_argument("--persist-dir",
                   default="chroma_db",
                   help="Path to ChromaDB on-disk storage")
    p.add_argument("--collection",
                   default="code_chunks",
                   help="Chroma collection name")
    p.add_argument("--model",
                   default="voyage-code-3",
                   help="Voyage embedding model")
    p.add_argument("--top-k", type=int, default=10,
                   help="How many nearest neighbors to fetch as context")
    args = p.parse_args()

    # 1) Voyage client (reads VOYAGE_API_KEY from env)
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # 2) Chroma persistent client
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"source": "code_chunks", "model": args.model}
    )

    # ensure output dir exists
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    with open(args.test_file, 'r', encoding='utf-8') as fin, \
         open(args.out_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            test = json.loads(line)
            prompt = test.get("prompt", "")
            meta = test.get("metadata", {})
            task_id = meta.get("task_id", "")
            fpath_tuple = meta.get("fpath_tuple")  # should be a list or "/"-joined string

            if not prompt or "/" not in task_id or not fpath_tuple:
                continue

            repo = task_id.split("/", 1)[0]

            # 3) embed the prompt as a "query"
            resp = vo.embed(
                [prompt],
                model=args.model,
                input_type="query",
                truncation=True
            )
            query_emb = resp.embeddings[0]

            # 4) fetch 2× top_k, filtered by repo
            fetch_n = args.top_k * 2
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=fetch_n,
                where={"repository": repo}
            )
            cand_ids     = results["ids"][0]
            cand_metas   = results["metadatas"][0]

            # 5) post-filter out same-file hits
            context_ids = []
            for cid, md in zip(cand_ids, cand_metas):
                # md["fpath_tuple"] was stored as slash-joined string
                if md["fpath_tuple"] != ("/".join(fpath_tuple) if isinstance(fpath_tuple, list) else fpath_tuple):
                    context_ids.append(cid)
                    if len(context_ids) >= args.top_k:
                        break

            # 6) augment record
            # test["embedding"] = query_emb
            test["context"]   = context_ids

            # write JSONL
            fout.write(json.dumps(test) + "\n")


    print("✅ Done! Wrote with embeddings & context to", args.out_file)


if __name__ == "__main__":
    main()
