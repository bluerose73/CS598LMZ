import os
import json
import zipfile
import argparse

# Recommended file extentions:
# .py, .java, .md

def unzip_repo(zip_path, extract_to):
    """Unzips the repository archive if not already extracted."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

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
    # For example, with window_size=20 and slice_size=2, step=10.
    step = window_size // slice_size if window_size // slice_size > 0 else 1
    chunks = []
    for start in range(0, n, step):
        end = min(n, start + window_size)
        if end - start > 0:
            chunk_text = "".join(lines[start:end])
            chunks.append((chunk_text, start, end))
    return chunks

def process_repository(repo_dir, repository_name, output_json, window_size, slice_size, allowed_extensions=None):
    """
    Walks through all files in the repository directory, splits text files into overlapping code chunks,
    and writes them as JSON Lines.
    Also writes a separate text file listing the processed file paths.
    
    Args:
        repo_dir: Directory containing the repository
        repository_name: Name of the repository
        output_json: Path to write the output JSONL file
        window_size: Size of the sliding window
        slice_size: Size of the slice for sliding
        allowed_extensions: List of file extensions to process (e.g. ['.py', '.java']). If None, defaults to ['.py', '.java', '.md']
    """
    
    code_chunks = []
    processed_files = []
    chunk_counter = 0
    for root, _, files in os.walk(repo_dir):
        for file in files:
            # Only process allowed file types
            if not file.lower().endswith(tuple(allowed_extensions)):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1)  # Attempt to read a little text.
            except Exception as e:
                continue  # Skip files that cannot be read (likely binary).

            rel_path = os.path.relpath(file_path, repo_dir)
            processed_files.append(rel_path)
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

    # Save processed file list in the same directory as output_json.
    output_txt = os.path.join(os.path.dirname(output_json), f"processed-files_{repository_name}.txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        for path in processed_files:
            f.write(path + "\n")
    print(f"List of processed files written to {output_txt}")

def process_all_repos(repo_names, repos_dir, data_schema_dir, window_size, slice_size, allowed_extensions=None):
    """
    For each repository, unzips it (if needed) and processes its code files into a code-chunks JSONL file.
    
    Args:
        repo_names: List of repository names to process
        repos_dir: Directory containing the zip files
        data_schema_dir: Output directory for code chunks
        window_size: Size of the sliding window
        slice_size: Size of the slice for sliding
        allowed_extensions: List of file extensions to process (e.g. ['.py', '.java']). If None, defaults to ['.py', '.java', '.md']
    """
    os.makedirs(data_schema_dir, exist_ok=True)
    
    for repo in repo_names:
        zip_path = os.path.join(repos_dir, f"{repo}.zip")
        extract_dir = os.path.join(repos_dir, repo)
        output_json = os.path.join(data_schema_dir, f"code-chunks_{repo}.jsonl")
        if not os.path.exists(extract_dir):
            unzip_repo(zip_path, extract_dir)
        else:
            print(f"Repository '{repo}' already extracted at {extract_dir}")
        process_repository(extract_dir, repo, output_json, window_size, slice_size, allowed_extensions)

def main():
    parser = argparse.ArgumentParser(description="Process repositories into code-chunks files.")
    parser.add_argument("--extensions", type=str, default=".py,.java,.md",
                       help="Comma-separated list of file extensions to process (e.g. '.py,.java,.md')")
    args = parser.parse_args()

    # Parse the extensions
    allowed_extensions = [ext.strip() for ext in args.extensions.split(',') if ext.strip()]
    # Ensure all extensions start with a dot
    allowed_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in allowed_extensions]

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
    
    repos_dir = "repositories"
    data_schema_dir = "code-chunks"
    window_size = 20
    slice_size = 2

    process_all_repos(
        repo_names, 
        repos_dir, 
        data_schema_dir, 
        window_size, 
        slice_size, 
        allowed_extensions=allowed_extensions
    )

if __name__ == '__main__':
    main()
