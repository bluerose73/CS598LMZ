#!/usr/bin/env python3
import os
import json
import argparse
from tree_sitter import Parser
from tree_sitter_languages import get_language

# ——— Configuration ———
PY_LANGUAGE = get_language('python')
JAVA_LANGUAGE = get_language('java')
LANGUAGE_MAP = {
    '.py': PY_LANGUAGE,
    '.java': JAVA_LANGUAGE,
}
NODE_TYPES = {
    'python': ['function_definition', 'class_definition'],
    'java': [
        'method_declaration',
        'class_declaration',
        'interface_declaration',
        'enum_declaration',
        'annotation_type_declaration',
    ],
}

def extract_tree_sitter_chunks(file_path, repo_name):
    """
    Parses a source file and extracts code chunks corresponding to functions, classes,
    interfaces, enums, and annotations—with any immediately preceding comments.
    Returns list of (full_snippet, start_line_no, end_line_no).
    """
    ext = os.path.splitext(file_path)[1].lower()
    language = LANGUAGE_MAP.get(ext)
    if not language:
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            src = f.read()
    except UnicodeDecodeError:
        print(f"Skipping (non-UTF8) {file_path}")
        return []

    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(src.encode('utf8'))
    root = tree.root_node
    lines = src.splitlines()
    chunks = []
    stack = [root]

    while stack:
        node = stack.pop()
        lang_name = language.name

        if node.type in NODE_TYPES.get(lang_name, []):
            # 1) Node’s own bounds
            start_row, _ = node.start_point
            end_row,   _ = node.end_point

            # 2) Scan *upwards* to include preceding comment lines
            comment_start = start_row
            i = start_row - 1
            while i >= 0:
                raw = lines[i]
                s = raw.lstrip()
                if lang_name == 'python' and s.startswith('#'):
                    comment_start = i
                elif lang_name == 'java' and s.startswith('//'):
                    comment_start = i
                elif lang_name == 'java' and '*/' in s:
                    # block comment end → scan until /* 
                    comment_start = i
                    i -= 1
                    while i >= 0:
                        raw2 = lines[i]
                        comment_start = i
                        if '/*' in raw2:
                            break
                        i -= 1
                    break
                else:
                    break
                i -= 1

            # 3) Slice lines from comment_start through end_row
            snippet_lines = lines[comment_start : end_row + 1]
            # reconstruct the snippet
            full_snippet = "\n".join(snippet_lines)

            # 4) Record it, storing the adjusted start_line_no
            chunks.append((full_snippet, comment_start, end_row))

        # push children to continue traversal
        stack.extend(node.children)

    return chunks


def process_repository_with_treesitter(repo_dir, repo_name, output_json):
    """
    Walks through files in repo_dir, extracts Tree-Sitter chunks (with comments),
    and writes them to output_json as JSONL. Also prints counts.
    """
    all_chunks = []
    processed_files = []
    chunk_counter = 0

    for root, _, files in os.walk(repo_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in LANGUAGE_MAP:
                continue

            path = os.path.join(root, fn)
            rel = os.path.relpath(path, repo_dir)
            processed_files.append(rel)

            print(f"Processing file: {rel}")
            for snippet, start, end in extract_tree_sitter_chunks(path, repo_name):
                fpath_tuple = [repo_name] + rel.split(os.sep)
                record = {
                    "id": chunk_counter,
                    "code": snippet,
                    "metadata": {
                        "repository": repo_name,
                        "fpath_tuple": fpath_tuple,
                        "start_line_no": start,
                        "end_line_no": end,
                    }
                }
                all_chunks.append(record)
                chunk_counter += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as out:
        for rec in all_chunks:
            out.write(json.dumps(rec) + "\n")

    print(f"[Tree-sitter] {repo_name}: {len(all_chunks)} chunks → {output_json}")
    # Optionally save processed files list
    txt_path = os.path.join(os.path.dirname(output_json), f"processed-files_{repo_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as ft:
        for p in processed_files:
            ft.write(p + "\n")
    print(f"[Tree-sitter] Processed files list → {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Build smart code chunks via Tree-Sitter")
    parser.add_argument("--repos-dir", default="repositories",
                        help="Directory with one subfolder per repo")
    parser.add_argument("--out-dir", default="code-chunks-smart",
                        help="Where to write <repo>.jsonl files")
    args = parser.parse_args()

    for repo_name in sorted(os.listdir(args.repos_dir)):
        repo_path = os.path.join(args.repos_dir, repo_name)
        if not os.path.isdir(repo_path):
            continue
        print(f"[Tree-sitter] Processing repository: {repo_name}")
        dest = os.path.join(args.out_dir, f"code-chunks_{repo_name}.jsonl")
        process_repository_with_treesitter(repo_path, repo_name, dest)


if __name__ == "__main__":
    main()
