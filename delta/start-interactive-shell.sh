# Use this command to start an interactive job on the server
# This starts an interactive shell with 1 A40 GPU, good for experimenting with code

srun --account=becw-delta-gpu --partition=gpuA40x4 --time=1200 \
  --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --mem=50g \
  --pty bash