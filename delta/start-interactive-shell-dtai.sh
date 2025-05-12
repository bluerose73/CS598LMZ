# Use this command to start an interactive job on Delta AI.
# Used for training

srun --account=becw-dtai-gh --partition=ghx4 --time=1200 \
  --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 \
  --cpus-per-task=72 --mem=100g \
  --pty bash

srun --account=becw-dtai-gh --partition=ghx4 --time=1200 \
  --nodes=1 --gpus-per-node=2 --ntasks-per-node=1 \
  --cpus-per-task=144 --mem=200g \
  --pty bash

srun --account=becw-dtai-gh --partition=ghx4 --time=1200 \
  --nodes=1 --gpus-per-node=4 --ntasks-per-node=1 \
  --cpus-per-task=288 --mem=400g \
  --pty bash