import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the axes labels
input_tokens = [256, 1024, 4096, 16384]
new_tokens = [16, 32, 64, 128]
matrix = np.full((len(input_tokens), len(new_tokens)), np.nan)  # Initialize with NaNs

# Parse the input file
with open('qwen2-prefill-output.txt', 'r') as file:
    lines = file.readlines()

input_tok = None
new_tok = None

for line in lines:
    if "Input tokens" in line:
        input_tok = int(re.search(r"Input tokens: (\d+)", line).group(1))
        new_tok = int(re.search(r"New tokens: (\d+)", line).group(1))
    elif "Percent Prefill" in line:
        percent = float(re.search(r"Percent Prefill: ([\d.]+)%", line).group(1))
        if input_tok in input_tokens and new_tok in new_tokens:
            i = input_tokens.index(input_tok)
            j = new_tokens.index(new_tok)
            matrix[i, j] = percent

# Plot the heatmap and save it
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=new_tokens, yticklabels=input_tokens)
plt.xlabel("New Tokens")
plt.ylabel("Input Tokens")
plt.title("Percent Prefill Heatmap")
plt.tight_layout()
plt.savefig("percent_prefill_heatmap.png", dpi=300)
plt.close()

# Plot the line chart
plt.figure(figsize=(8, 6))
for j, n_tok in enumerate(new_tokens):
    plt.plot(input_tokens, matrix[:, j], marker='o', label=f"{n_tok} new tokens")

plt.xscale('log')
plt.xlabel("Input Tokens (log scale)")
plt.ylabel("Percent Prefill")
plt.title("Percent Prefill vs. Input Tokens")
plt.legend(title="New Tokens")
plt.grid(True)
plt.tight_layout()
plt.savefig("percent_prefill_line_chart.png", dpi=300)
plt.close()
