# Generation

run inference for decoder-only and FiD models.

The command is (suppose you are at the repository's root directory)

```bash
python -m generation <config-file-name.yaml>
```

See [generation config example](generation-config.yaml) on how to write a config file.


## Setting Up the Environment on Delta  

1. **SSH into a Delta login node:**  
   Open a terminal and connect to a Delta login node using SSH.  

2. **Load the Anaconda module:**  
   Run the following command to load Anaconda:  
   ```bash
   module load anaconda3_gpu
   ```
   Then, initialize Conda for your shell:  
   ```bash
   conda init
   ```
   Close and reopen your terminal. If the setup is successful, you should see `(base)` in your terminal prompt.  

   This step only needs to be performed once. On future logins, the Anaconda module should load automatically.  

3. **Create a new Conda environment:**  
   Run the following commands to create and activate a new Conda environment named `fid`:  
   ```bash
   conda create -n fid python=3.12
   conda activate fid
   ```

4. **Install dependencies using `pip`:**  
   Ensure the `fid` environment is activated, then install dependencies by running:  
   ```bash
   pip install -r generation/requirements.txt
   ```

---

## Running Inference on Delta  

1. **SSH into a Delta login node.**  

2. **Start an interactive session:**  
   Run the script below to start an interactive shell:  
   ```bash
   delta/start-interactive-shell.sh
   ```
   Your terminal prompt should change from `username@dt-login01` to something like `username@gpu027`, indicating you are now on a GPU node.  

3. **Activate the Conda environment:**  
   ```bash
   conda activate fid
   ```

4. **Run the inference script:**  
   Execute the following command, replacing `<config-file-name.yaml>` with your actual configuration file:  
   ```bash
   python -m generation <config-file-name.yaml>
   ```  
