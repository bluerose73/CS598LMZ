`process_repo_training.py`

Takes in unzipped repositories and returns 2 things:

1. train/
- For each repo_name, output a repo_name.jsonl file with full prompt and empty context

2. training-code-chunks/
- For each repo_name, output a .jsonl file with code chunks

You can change the window size and the slice:

```
python3 process_repo_training.py --window-size 20 --slice-size 2
```

`augment.py`

You can generate relevent BM25 context, which get outputted to:

train-with-context/

This takes in both train/ and training-code-chunks/ matching on repo_name and computes BM25 scores (with current training file removed to avoid double context when training).