import os
import json
import zipfile

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
    # Determine the step. Here we use: step = window_size // slice_size (or at least 1)
    step = window_size // slice_size if window_size // slice_size > 0 else 1
    chunks = []
    for start in range(0, n, step):
        end = min(n, start + window_size)
        # Only add non-empty chunks
        if end - start > 0:
            chunk_text = "".join(lines[start:end])
            chunks.append((chunk_text, start, end))
    return chunks

def process_repository(repo_dir, repository_name, output_json, window_size, slice_size):
    """
    Iterates through all files in the unzipped repository and builds overlapping code chunks.
    Each chunk is stored as a JSON object with a unique id and metadata.
    
    Additionally, records the relative paths of processed (text) files into a list.
    """
    code_chunks = []
    processed_files = []  # list of relative file paths that were processed
    chunk_counter = 0
    # Walk through the repository directory recursively.
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Try reading the file as text; skip binary files.
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Attempt to read one character.
                    f.read(1)
            except Exception as e:
                # Skip files that raise an exception (likely binary files)
                continue

            # Record the file as processed.
            rel_path = os.path.relpath(file_path, repo_dir)
            processed_files.append(rel_path)
            
            # Build fpath_tuple relative to repo_dir. Prepend repository_name.
            fpath_tuple = [repository_name] + rel_path.split(os.sep)
            
            # Create chunks from this file.
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

    # Write the code chunks as JSON Lines.
    with open(output_json, 'w', encoding='utf-8') as f:
        for chunk in code_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Processed repository '{repository_name}': created {len(code_chunks)} code chunks.")

    # Also output the list of processed file paths to a .txt file.
    output_txt = output_json.replace("code-chunks", "processed-files").replace(".jsonl", ".txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        for file_path in processed_files:
            f.write(file_path + "\n")
    print(f"List of processed files written to {output_txt}")

def main():
    # Path to the repository zip archive and where to extract it.
    base_repo_zip = "/u/dylandunham/CS598LMZ/prompt-builder/repositories/Aelysium-Group_rusty-connector.zip"
    extract_dir = "/u/dylandunham/CS598LMZ/prompt-builder/repositories/Aelysium-Group_rusty-connector"
    # Output file for the code chunks (JSON Lines format)
    output_json = "/u/dylandunham/CS598LMZ/prompt-builder/data-schema/code-chunks_Aelysium-Group_rusty-connector.jsonl"
    window_size = 20
    slice_size = 2

    # Unzip the repository if it has not already been extracted.
    if not os.path.exists(extract_dir):
        unzip_repo(base_repo_zip, extract_dir)
    else:
        print(f"Repository already extracted at {extract_dir}")
    
    # Process the extracted repository files to create code chunks and record processed file paths.
    repository_name = "Aelysium-Group_rusty-connector"
    process_repository(extract_dir, repository_name, output_json, window_size, slice_size)

if __name__ == '__main__':
    main()
