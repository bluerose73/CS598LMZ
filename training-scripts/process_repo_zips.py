import os
import zipfile
import shutil
import argparse
from tqdm import tqdm
from typing import List
from trainer.types import CodeChunk, CodeToComplete
import multiprocessing as mp


# -------------------------------------------------------------------
# Retriever base class and a path distance implementation

class Retriever:
    def retrieve(self, target_chunk: CodeChunk, code_chunks: List[CodeChunk], n_context: int) -> List[int]:
        raise NotImplementedError("This method should be overridden by subclasses.")


class PathDistanceRetriever(Retriever):
    def retrieve(self, target_chunk: CodeChunk, code_chunks: List[CodeChunk], n_context: int) -> List[int]:
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
            dist = path_distance(target_chunk.fpath_tuple, chunk.fpath_tuple)
            distances.append((dist, chunk.id))
        # Sort by distance: smaller distance means more similar path
        distances.sort(key=lambda x: x[0])
        # Return the top n_context code chunk ids
        return [chunk_id for _, chunk_id in distances[:n_context]]


# -------------------------------------------------------------------
# Main preprocessor function

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess zipped GitHub repositories into training data for a code language model."
    )
    parser.add_argument('--data-dir', default=None,
                        help="Directory containing zip files of GitHub repositories. Naming: <author>_<repositoryname>.zip")
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
    parser.add_argument('--unzipped-dir', default=None,
                        help="Directory containing unzipped GitHub repositories. If provided, these will also be processed.")
    args = parser.parse_args()

    # Ensure at least one of --data-dir or --unzipped-dir is provided.
    if not args.data_dir and not args.unzipped_dir:
        raise ValueError("At least one of --data-dir or --unzipped-dir must be provided.")

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

    # List all zip files in the data directory if provided.
    zip_files = []
    if args.data_dir:
        zip_files = [f for f in os.listdir(args.data_dir) if f.endswith('.zip')]

    # If unzipped-dir is provided, list all directories in it.
    unzipped_dirs = []
    if args.unzipped_dir:
        unzipped_dirs = [os.path.join(args.unzipped_dir, d) for d in os.listdir(args.unzipped_dir)
                         if os.path.isdir(os.path.join(args.unzipped_dir, d))]

    # Process each zip file with a progress bar.
    for zip_file in tqdm(zip_files, desc="Processing zipped repos"):
        zip_path = os.path.join(args.data_dir, zip_file)
        # Extract the zip file.
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(args.tmp_dir)
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")
            continue

        # Determine the repository name based on the root directory of the unzipped content.
        try:
            extracted_dirs = os.listdir(args.tmp_dir)
            repository = next(
                (d for d in extracted_dirs if os.path.isdir(os.path.join(args.tmp_dir, d))),
                None
            )
            if not repository:
                print(f"Skipping {zip_file} because no valid root directory was found after extraction.")
                continue
        except Exception as e:
            print(f"Error determining repository name for {zip_file}: {e}")
            continue

        repo_root = os.path.join(args.tmp_dir, repository)
        # Process the repository.
        process_repository(repo_root, repository, retriever, allowed_extensions, args)

        # Clean up: remove the unzipped repository directory.
        shutil.rmtree(repo_root, ignore_errors=True)

    # -------------------------------------------------------------------
    # Multiprocessing for unzipped repositories
    # -------------------------------------------------------------------
    if unzipped_dirs:
        # Prepare argument tuples for worker processes.
        mp_params = [
            (repo_root, retriever, allowed_extensions, args)
            for repo_root in unzipped_dirs
        ]
        with mp.Pool(processes=min(mp.cpu_count(), len(mp_params))) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_unzipped_repo, mp_params),
                total=len(mp_params),
                desc="Processing unzipped repos (mp)"
            ):
                pass


def process_repository(repo_root, repository, retriever, allowed_extensions, args):
    code_chunks = []
    chunk_id = 0
    # Walk through repository files and build code chunks.
    for root, _, files in os.walk(repo_root):
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
    code_to_complete_list = []
    for chunk in code_chunks:
        # Retrieve context (list of code chunk ids excluding itself).
        context = retriever.retrieve(chunk, code_chunks, args.n_context)
        code_to_complete = CodeToComplete(
            code=chunk.code,
            context=context,
            repository=repository,
            fpath_tuple=chunk.fpath_tuple,
            metadata={}
        )
        code_to_complete_list.append(code_to_complete)

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


# -------------------------------------------------------------------
# Helper for multiprocessing (must be top-level picklable function)
# -------------------------------------------------------------------
def _process_unzipped_repo(params):
    repo_root, retriever, allowed_extensions, args = params
    repository = os.path.basename(repo_root)
    process_repository(repo_root, repository, retriever, allowed_extensions, args)


if __name__ == "__main__":
    main()
