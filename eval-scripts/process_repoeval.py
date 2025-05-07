# Perform RAG and build repoeval-updated test dataset

import os
import zipfile
import shutil
import argparse
import json
from tqdm import tqdm
from fid.data_util.types import CodeChunk, CodeToComplete
from fid.data_util.retriever import PathDistanceRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess zipped GitHub repositories into training data for a code language model."
    )
    parser.add_argument('--data-dir', required=True,
                        help="Directory containing zip files of GitHub repositories. Naming: <author>_<repositoryname>.zip")
    parser.add_argument('--jsonl-path', required=True,
                        help="Path for the input line completion jsonl file from RepoEval-Updated.")
    parser.add_argument('--tmp-dir', default='./train-data-builder-tmp',
                        help="Temporary directory to store unzipped repositories. Default: ./train-data-builder-tmp")
    parser.add_argument('--retriever', default='path-distance',
                        help="Retriever to use for context (now only supports 'path-distance').")
    parser.add_argument('--n-context', type=int, default=12,
                        help="Number of retrieved context chunks. Default: 12")
    parser.add_argument('--out-code-chunk-dir', required=True,
                        help="Directory to store output code chunks in jsonl format.")
    parser.add_argument('--out-code-to-complete-dir', required=True,
                        help="Directory to store output code-to-complete in jsonl format.")
    parser.add_argument('--allowed-extensions', type=str, default='.py,.java,.md',
                        help="Comma-separated list of file extensions to process (e.g. '.py,.java,.md'). Use '*' to allow all extensions. Default: '.py,.java,.md'")
    args = parser.parse_args()


    # Create directories if they do not exist.
    if args.tmp_dir and os.path.exists(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_code_chunk_dir, exist_ok=True)
    os.makedirs(args.out_code_to_complete_dir, exist_ok=True)

    # Parse allowed extensions.
    if args.allowed_extensions == '*':
        allowed_extensions = None  # Allow all extensions
        print("All file extensions are allowed.")
    else:
        allowed_extensions = args.allowed_extensions.split(',')
        allowed_extensions = [ext.strip() for ext in allowed_extensions if ext.strip()]

    # For now, we only support path-distance retriever.
    if args.retriever != "path-distance":
        raise ValueError("Only 'path-distance' retriever is currently supported.")
    retriever = PathDistanceRetriever()


    # Read jsonl file
    code_to_complete_dict = {}  # repo_name -> list of code_to_complete
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            repository = data['metadata']['task_id'].split('/')[0]
            code_to_complete = CodeToComplete(
                code=data['prompt'],
                context=[],
                repository=repository,
                fpath_tuple=data['metadata']['fpath_tuple'][1:],
                metadata=data['metadata']
            )
            del code_to_complete.metadata['fpath_tuple']
            if repository not in code_to_complete_dict:
                code_to_complete_dict[repository] = []
            code_to_complete_dict[repository].append(code_to_complete)

    # RAG
    for repo_name, code_to_complete_list in code_to_complete_dict.items():
        print(f"Processing repository: {repo_name}")
        zip_file = os.path.join(args.data_dir, f"{repo_name}.zip")
        if not os.path.exists(zip_file):
            raise ValueError(f"Zip file {zip_file} does not exist.")
        
        # Extract the zip file.
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(args.tmp_dir)
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")
            continue

        subdirs = os.listdir(args.tmp_dir)
        if len(subdirs) != 2:
            raise ValueError(f"Expected 2 subdirectories in {args.tmp_dir}, found {len(subdirs)}. Please check the zip file structure.")
        for subdir in subdirs:
            if subdir != '__MACOSX':
                repo_root = os.path.join(args.tmp_dir, subdir)
                
        # repo_root = os.path.join(args.tmp_dir, repo_name)
        # if not os.path.exists(repo_root):
        #     # Attempt to extract repo_name after the first underscore
        #     print(f"Repository root {repo_root} not found. Trying to extract after the first underscore.")
        #     alt_repo_name = repo_name.split('_', 1)[-1] if '_' in repo_name else None
        #     if alt_repo_name:
        #         repo_root = os.path.join(args.tmp_dir, alt_repo_name)
        #     if not os.path.exists(repo_root):
        #         print(f"Repository root {repo_root} still not found. Trying lowercase.")
        #         alt_repo_name = alt_repo_name.lower()
        #         repo_root = os.path.join(args.tmp_dir, alt_repo_name)
        #         if not os.path.exists(repo_root):
        #             raise ValueError(f"Repository {repo_root} does not exist after extraction.")
        
        # Process the repository.
        process_repository(repo_root, repo_name, code_to_complete_list, retriever, allowed_extensions, args)

        # Clean up: remove the unzipped repository directory.
        shutil.rmtree(repo_root, ignore_errors=True)


def process_repository(repo_root, repository, code_to_complete_list, retriever, allowed_extensions, args):
    code_chunks = []
    chunk_id = 0
    # Walk through repository files and build code chunks.
    for root, _, files in tqdm(os.walk(repo_root), desc="Processing code chunks", unit="file"):
        for file in files:
            # Check if the file has an allowed extension.
            if allowed_extensions and not any(file.endswith(ext) for ext in allowed_extensions):
                continue
            file_path = os.path.join(root, file)
            # Get the file path relative to the repository root and split into tuple.
            rel_path = os.path.relpath(file_path, repo_root)
            fpath_tuple = rel_path.split(os.sep)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            code_chunk = CodeChunk(
                code=code,
                id=chunk_id,
                repository=repository,
                fpath_tuple=fpath_tuple,
                metadata={}
            )
            code_chunks.append(code_chunk)
            chunk_id += 1

    # Now produce code-to-complete records.
    for code_to_complete in tqdm(code_to_complete_list, desc="Retrieval", unit="record"):
        # Retrieve context (list of code chunk ids excluding itself).
        key_chunk = CodeChunk(
            code=code_to_complete.code,
            id=-1,
            repository=repository,
            fpath_tuple=code_to_complete.fpath_tuple,
        )
        context = retriever.retrieve(key_chunk, code_chunks, args.n_context, cross_file=True)
        code_to_complete.context = context

    # Write outputs in jsonl format.
    repo_name = repository.replace('/', '_')  # Ensure valid filename
    code_chunk_filename = f"code-chunks_{repo_name}.jsonl"
    code_to_complete_filename = f"code-to-complete_{repo_name}.jsonl"

    out_chunk_path = os.path.join(args.out_code_chunk_dir, code_chunk_filename)
    out_to_complete_path = os.path.join(args.out_code_to_complete_dir, code_to_complete_filename)

    with open(out_chunk_path, 'w', encoding='utf-8') as f_out:
        for chunk in code_chunks:
            f_out.write(chunk.model_dump_json() + "\n")

    with open(out_to_complete_path, 'w', encoding='utf-8') as f_out:
        for item in code_to_complete_list:
            f_out.write(item.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
