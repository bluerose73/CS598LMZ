# Training Fusion-In Decoder for Fast Long-Context Code Completion

Course project for CS598LMZ: Software Quality Assurance with Generative AI at UIUC

Shengjie Ma, Dylan Dunham, Zhijie Chen, Yifan Shen


Large language model based code completion tools have transformed software development workflows, offering productivity boosts through intelligent autocompletion. However, these tools face a trade-off between leveraging extensive repository-wide context and maintaining low latency. While decoder-only LLMs such as Qwen2.5-Coder-3B deliver strong performance, inference with long context inevitably increases the latency, due to expensive prefill computations. In addition, their reliance on causal attention limits KV cache reuse. To address this, we extend a decoder-only model into a Fusion-in Decoder. Our approach includes training a lightweight encoder and cross-attention layers together with a frozen decoder, allowing code chunks to be encoded in parallel. Encoder hidden states can also be cached and reused. Experiments on RepoEval-Updated demonstrate that our method significantly reduces prefill latency on long prompts, while showing a completion accuracy gain compared with the in-file baseline.


## Reproducing the Result

1. Python environment setup


   The code is tested only with Python 3.12 and linux OS.

   Install dependencies by running

   ```bash
   pip install -r requirements.txt
   ```

   Add the repository root directory to your python package search path

   ```bash
   export PYTHONPATH=path/to/bluerose73/cs598lmz
   ```

2. Download the FiD model checkpoint and tokenized dataset

   Please download in the following Google Drive link. (Google Apps @ Illinois account required.)
   https://drive.google.com/drive/folders/1fZxPQx8Lq6P8NyI0Zdl_1RCteRuwcqUK?usp=drive_link

   Unzip data.zip and put it under repository root directory. The path will look like this

   ```
   data/
   ├── repoeval-bm25      # Sliding-Window BM25
   ├── repoeval-smart     # TreeSitter-EmbedRAG
   └── repoeval-updated   # Truncation Path-Distance
   ```

   Download the FiD model checkpoint `*.ckpt` and put it in a directory you like.

3. Run generation

   Below is the script used for the RepoEval-Updated Java task, using Sliding-Window BM25 retriever. To run generation for other settings, please change the input and output dirs.

   ```bash
   # Run FiD generation.
   # ~10 minutes on 1xA100
   python eval-scripts/fid_generate_repoval.py --input_dir data/repoeval-bm25/python/tokenized/ --output_dir data/repoeval-bm25/python/completion --model_path path/to/checkpoint.ckpt

   # Run Qwen2.5-Coder-3B in-file and cross-file baselines generation
   # ~1 hour on 1xA100
   python eval-scripts/qwen2_generate_repoval.py --input_dir data/repoeval-bm25/python/tokenized/ --output_dir data/repoeval-bm25/python/completion
   ```

4. Calculate metrics

   ```bash
   python eval-scripts/eval_repoeval.py --language python --completion-path data/repoeval-bm25/python/completion/fid-copy-completion.jsonl

   python eval-scripts/eval_repoeval.py --language java --completion-path data/repoeval-bm25/java/completion/fid-copy-completion.jsonl
   ```

   This will report Exact Match and Edit Similarity.

## Running the Complete Inference Pipeline

The whole inference pipeline consist 4 stages.

1. Process (chunk) the repositories and augment prompts with retrieved cross-file context.

   See [prompt-builder/readme.md](prompt-builder/readme.md)

2. Tokenize the dataset

   Take Sliding-Window BM25 as an example, run the following script

   ```bash
   bash eval-scripts/tokenize_repoeval_bm25.sh
   ```

3. Run model generation
   
   In the Reproducing the Result section, you start from this step. See the Reproducing the Result section.

4. Calculate metrics

   See the Reproducing the Result section.

## Training the Model

The training scripts contain some hard-coded paths and configurations specifically for Delta / DeltaAI. Below is a high-level walkthrough on how to train the model.

1. Download an 16k subset of The Stack V2. You may want to edit the dataset save path in the code.

   ```bash
   python training-scripts/download-the-stack.py
   ```

2. Run retrieval using path-distance retriever, and then tokenize the dataset. You may want to edit the paths in the shell script and Python scripts.

   ```bash
   bash training-scripts/retrieve_and_tokenize_stack_20k.sh
   ```

3. Start training. You can now provide paths and model names as command-line arguments. For example:
   ```bash
   python training-scripts/train_on_the_stack_v2.py \
       --decoder_config_path ./fid/model/config.json \
       --encoder_model_name_or_path Qwen/Qwen2.5-Coder-0.5B \
       --decoder_model_name_or_path Qwen/Qwen2.5-Coder-3B \
       --wandb_log_dir ./wandb-logs \
       --tokenized_data_load_dir /path/to/your/tokenized_data
   ```

## Description of Sub-Directories

**Core Directories**

| folder           | description                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| fid              | a Python package including model architecture, training step, data utils, etc. Not directly runnable, but many scripts in this repository depends on this package. |
| prompt-builder   | build the prompt for repo-level code completion using RAG                                                                                                          |
| eval-scripts     | runnable python scripts for evaluation                                                                                                                             |
| training-scripts | runnable python scripts for training.                                                                                                                              |


**Secondary Directories**

| folder     | description                                    |
| ---------- | ---------------------------------------------- |
| delta      | scripts for running jobs in NCSA Delta cluster |
| analysis   | data & latency analysis and visualization      |
| test       | test scripts                                   |
| docs       | miscellaneous notes                            |