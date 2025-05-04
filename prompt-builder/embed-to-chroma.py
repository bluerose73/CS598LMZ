#!/usr/bin/env python3
"""
Embed code chunks with Voyage and store in a local ChromaDB,
but skip repos that have already been fully embedded.
Add & commit in batches to avoid Chroma's max-batch limit.
"""
import os
import json
import argparse
import voyageai
import chromadb
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Embed code chunks with Voyage and store in a local ChromaDB"
    )
    parser.add_argument(
        "--chunks-dir", default="code-chunks-smart",
        help="Directory containing code-chunks_<repo>.jsonl files"
    )
    parser.add_argument(
        "--persist-dir", default="chroma_db",
        help="Directory where Chroma will persist its data"
    )
    parser.add_argument(
        "--collection-name", default="code_chunks",
        help="Name of the Chroma collection to use/create"
    )
    parser.add_argument(
        "--model", default="voyage-code-3",
        choices=["voyage-3-large","voyage-3","voyage-3-lite","voyage-code-3","voyage-code-2"],
        help="Voyage embedding model to use"
    )
    args = parser.parse_args()

    # 1) Voyage client
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # 2) PersistentClient for on-disk storage (duckdb+parquet)
    client = chromadb.PersistentClient(path=args.persist_dir)

    # 3) get or create collection
    collection = client.get_or_create_collection(
        name=args.collection_name,
        metadata={"source": "code_chunks", "model": args.model}
    )

    # 4) process each repo
    for fname in sorted(os.listdir(args.chunks_dir)):
        if not fname.startswith("code-chunks_") or not fname.endswith(".jsonl"):
            continue

        repo = fname[len("code-chunks_"):-len(".jsonl")]
        path = os.path.join(args.chunks_dir, fname)

        # count how many code-chunks on disk
        with open(path, 'r', encoding='utf-8') as f:
            total_chunks = sum(1 for _ in f)

        # count how many embeddings already stored
        existing = len(collection.get(where={"repository": repo})["ids"])
        if existing >= total_chunks:
            print(f"→ {repo}: already have {existing}/{total_chunks}, skipping.")
            continue

        print(f"Embedding chunks for repo: {repo} ({existing}/{total_chunks} done)")

        # load everything
        texts, metadatas, ids = [], [], []
        with open(path, 'r', encoding='utf-8') as f:
            for obj in map(json.loads, f):
                cid = f"{repo}_{obj['id']}"
                texts.append(obj["code"])
                md = obj["metadata"].copy()
                md["fpath_tuple"] = "/".join(md["fpath_tuple"])
                metadatas.append(md)
                ids.append(cid)

        # embed & add in batches
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch_texts     = texts[i : i + batch_size]
            batch_ids       = ids[i : i + batch_size]
            batch_meta      = metadatas[i : i + batch_size]
            resp = vo.embed(
                batch_texts,
                model=args.model,
                input_type="document",
                truncation=True
            )
            collection.add(
                ids=batch_ids,
                embeddings=resp.embeddings,
                metadatas=batch_meta,
                documents=batch_texts
            )
            done = min(i + batch_size, len(texts))
            print(f"  Added {done}/{len(texts)} embeddings for {repo}")

    print("✔ All done. ChromaDB is at", args.persist_dir)


if __name__ == "__main__":
    main()
