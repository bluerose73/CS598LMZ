import os
import json
import zipfile
import argparse

# def unzip_repo(zip_path, extract_to):
#     """Unzips the repository archive if not already extracted."""
#     with zipfile.ZipFile(zip_path, 'r') as zf:
#         zf.extractall(extract_to)
#     print(f"Extracted {zip_path} to {extract_to}")

def read_full_file(file_path):
    """Reads the entire content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

def sliding_window_chunks(file_path, window_size, slice_size):
    """
    Reads the file and returns a list of tuples (chunk_text, start_line, end_line)
    using a sliding window approach.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return []
    
    n = len(lines)
    step = window_size // slice_size if window_size // slice_size > 0 else 1
    chunks = []
    for start in range(0, n, step):
        end = min(n, start + window_size)
        if end - start > 0:
            chunk_text = "".join(lines[start:end])
            chunks.append((chunk_text, start, end))
    return chunks

def process_repository_full(repo_dir, repository_name, output_json):
    """
    Walks through all files in the repository directory and writes the full file content as JSON Lines.
    
    Args:
        repo_dir: Directory containing the repository
        repository_name: Name of the repository
        output_json: Path to write the output JSONL file
    """
    
    file_entries = []
    processed_files = []
    file_counter = 0
    
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Try reading the file to check if it's readable text
                full_content = read_full_file(file_path)
                if full_content is None:
                    continue  # Skip if file couldn't be read
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

            rel_path = os.path.relpath(file_path, repo_dir)
            processed_files.append(rel_path)
            
            # Split path into components for fpath_tuple
            path_components = [repository_name] + rel_path.split(os.sep)
            
            # Create the file entry in the required format
            file_entry = {
                "prompt": full_content,
                "ground_truth": "",
                "file_path": rel_path,
                "metadata": {
                    "task_id": f"{repository_name}/{file_counter}",
                    "ground_truth": "", 
                    "fpath_tuple": path_components,
                    "context_start_lineno": 0,
                    "line_no": 0
                },
                "context": []
            }
            
            file_entries.append(file_entry)
            file_counter += 1

    # Write JSONL file
    with open(output_json, 'w', encoding='utf-8') as f:
        for entry in file_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Processed repository '{repository_name}': created {file_counter} full file entries.")

def process_repository_chunks(repo_dir, repository_name, output_json, window_size, slice_size):
    """
    Walks through all files in the repository directory, splits text files into overlapping code chunks,
    and writes them as JSON Lines.
    
    Args:
        repo_dir: Directory containing the repository
        repository_name: Name of the repository
        output_json: Path to write the output JSONL file
        window_size: Size of the sliding window
        slice_size: Size of the slice for sliding
    """
    
    code_chunks = []
    chunk_counter = 0
    for root, _, files in os.walk(repo_dir):
        for file in files:
            # Only process allowed file types

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1)  # Attempt to read a little text.
            except Exception as e:
                continue  # Skip files that cannot be read (likely binary).

            rel_path = os.path.relpath(file_path, repo_dir)
            # Create a file path tuple: [repository_name, subdirectory, file, ...]
            fpath_tuple = [repository_name] + rel_path.split(os.sep)
            chunks = sliding_window_chunks(file_path, window_size, slice_size)
            for chunk_text, start_line, end_line in chunks:
                chunk_obj = {
                    "code": chunk_text,
                    "id": chunk_counter,
                    "metadata": {
                        "repository": repository_name,
                        "fpath_tuple": fpath_tuple,
                        "start_line_no": start_line,
                        "end_line_no": end_line
                    }
                }
                code_chunks.append(chunk_obj)
                chunk_counter += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        for chunk in code_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Processed repository '{repository_name}': created {len(code_chunks)} code chunks.")

def process_all_repos(repo_names, repos_dir, full_dir, chunk_dir, window_size, slice_size):
    """
    Processes all repository directories in the given list.
    
    Args:
        repo_names: List of repository names to process
        repos_dir: Directory containing the repositories
        full_dir: Output directory for full file entries
        chunk_dir: Output directory for code chunks
        window_size: Size of the sliding window for chunking
        slice_size: Size of the slice for sliding
    """
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(chunk_dir, exist_ok=True)
    
    for repo in repo_names:
        repo_dir = os.path.join(repos_dir, repo)
        
        if not os.path.exists(repo_dir):
            print(f"Warning: Repository '{repo}' not found at {repo_dir}")
            continue
        
        # Process full file entries
        full_output_json = os.path.join(full_dir, f"{repo}.jsonl")
        process_repository_full(repo_dir, repo, full_output_json)
        
        # Process code chunks
        chunk_output_json = os.path.join(chunk_dir, f"{repo}.jsonl")
        process_repository_chunks(repo_dir, repo, chunk_output_json, window_size, slice_size)

def main():
    parser = argparse.ArgumentParser(description="Process repositories into full file entries and code chunks.")
    parser.add_argument("--repos-dir", type=str, default=".",
                      help="Directory containing repository directories")
    parser.add_argument("--full-dir", type=str, default="train",
                      help="Directory to save the full file entries")
    parser.add_argument("--chunk-dir", type=str, default="training-code-chunks",
                      help="Directory to save the code chunks")
    parser.add_argument("--window-size", type=int, default=20,
                      help="Size of the sliding window for chunking")
    parser.add_argument("--slice-size", type=int, default=2,
                      help="Size of the slice for sliding")
    args = parser.parse_args()

    # Get all directories in the current path (excluding hidden directories and output dirs)
    repo_names = []
    for item in os.listdir(args.repos_dir):
        item_path = os.path.join(args.repos_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item != args.full_dir and item != args.chunk_dir:
            repo_names.append(item)
    
    print(f"Found {len(repo_names)} repositories to process")
    
    process_all_repos(
        repo_names, 
        args.repos_dir, 
        args.full_dir, 
        args.chunk_dir,
        args.window_size,
        args.slice_size
    )

if __name__ == '__main__':
    main()