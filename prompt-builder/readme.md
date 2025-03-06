# Prompt Builder

A tool to process code repositories and augment prompts with relevant code context.

```
├── repositories/          # Input zipped repos
├── code-chunks/          # Processed code chunks
├── augmented-prompts/    # Output augmented prompts
└── RepoEval-Updated/     # Test files
```

1. **Process Repositories**

Creates `code-chunks/`

```
python process_repos.py
```

2. **Augment Prompts**
Uses BM25 to find relevant code chunks for each prompt and adds them as context. 

Outputs to `augmented-prompts/`.

```
python augment_prompts.py
```
