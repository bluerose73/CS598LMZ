#!/usr/bin/env python3
"""
Transforms vector-based context files from RepoEval-Updated-Context into the required schema,
organizing them by repository and saving to augmented-prompts-smart.
"""
import os
import json
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

def transform_context_ids(context_ids):
    """
    Transform context IDs by removing the repo name prefix.
    Example: "repo_name_123" -> 123
    """
    transformed = []
    for cid in context_ids:
        # Split by underscore and take the last part as the numeric ID
        parts = cid.split('_')
        if len(parts) > 1:
            try:
                transformed.append(int(parts[-1]))
            except ValueError:
                raise ValueError(f"Failed to convert context ID: {cid}. Expected format: 'repo_name_numeric_id'")
    return transformed

def process_file(input_file, output_dir):
    """
    Process a single input file from RepoEval-Updated-Context and split it by repo.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to write output files
    """
    # Extract test type and language from filename
    filename = os.path.basename(input_file)
    test_type, language = filename.replace('.test.jsonl', '').split('.')
    
    # Load the input file
    print(f"Loading {input_file}...")
    prompts = load_jsonl(input_file)
    
    # Group prompts by repository
    repo_prompts = {}
    for prompt in prompts:
        task_id = prompt.get("metadata", {}).get("task_id", "")
        if "/" not in task_id:
            continue
        
        repo = task_id.split("/", 1)[0]
        if repo not in repo_prompts:
            repo_prompts[repo] = []
        
        # Transform the context IDs
        if "context" in prompt:
            prompt["context"] = transform_context_ids(prompt["context"])
        
        repo_prompts[repo].append(prompt)
    
    # Write output files by repository
    for repo, prompts in repo_prompts.items():
        output_file = os.path.join(
            output_dir, 
            f"unfinished-code-w-context_{repo}_{test_type}.{language}_vector.jsonl"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        dump_jsonl(prompts, output_file)
        print(f"Wrote {len(prompts)} prompts to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Transform vector-based context files to match the required schema."
    )
    parser.add_argument(
        "--input-dir", 
        default="RepoEval-Updated-Context",
        help="Directory containing vector-based context files"
    )
    parser.add_argument(
        "--output-dir", 
        default="augmented-prompts-smart",
        help="Directory to write transformed files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.test.jsonl'):
            input_file = os.path.join(args.input_dir, filename)
            process_file(input_file, args.output_dir)

if __name__ == "__main__":
    main()