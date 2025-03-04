# Use this command to start an interactive job on the server
# This starts an interactive shell with 1 A40 GPU, good for experimenting with code
# The job will shutdown after 60 minutes

srun --account=becw-delta-gpu --partition=gpuA40x4-interactive --time=60 \
  --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --mem=50g --constraint="projects" \
  --pty bash