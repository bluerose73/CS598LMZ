python training-scripts/process_repo_zips.py \
    --allowed-extensions '*' \
    --unzipped-dir /work/nvme/becw/sma2/the-stack-v2-20k/repos \
    --out-code-chunk-dir /work/nvme/becw/sma2/the-stack-v2-20k/code-chunks \
    --out-code-to-complete-dir /work/nvme/becw/sma2/the-stack-v2-20k/code-to-complete

python training-scripts/tokenize_the_stack.py