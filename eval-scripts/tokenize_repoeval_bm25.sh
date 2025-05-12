python eval-scripts/tokenize_test_dataset.py \
    --code_chunks_dir data/repoeval-bm25/python/code-chunks/ \
    --code_to_complete_dir data/repoeval-bm25/python/code-to-complete/ \
    --tokenized_data_save_dir data/repoeval-bm25/python/tokenized \
    --requires_parser \
    --code_to_complete_filename_prefix unfinished-code-w-context_

python eval-scripts/tokenize_test_dataset.py \
    --code_chunks_dir data/repoeval-bm25/java/code-chunks/ \
    --code_to_complete_dir data/repoeval-bm25/java/code-to-complete/ \
    --tokenized_data_save_dir data/repoeval-bm25/java/tokenized \
    --requires_parser \
    --code_to_complete_filename_prefix unfinished-code-w-context_