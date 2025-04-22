# How much GPU memory does the encoder-decoder model use?

import lightning as L
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
import torch
from torch.cuda import OutOfMemoryError

# TF32 tensor cores
torch.set_float32_matmul_precision("high")

torch.cuda.memory._record_memory_history()

# Load models
encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

print(f"peak memory allocated for models: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Create dummy input
input_length = 2048
batch_size = 8

dummy_input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, input_length),
    dtype=torch.long,
    device=encoder.device,
)

# Run encoder forward pass with grad

encoder_outputs = encoder(input_ids=dummy_input_ids)

print(f"peak memory allocated after encoder forward: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Save CUDA memory snapshot
torch.cuda.memory._dump_snapshot("encoder_inference_snapshot.pickle")
print("CUDA memory snapshot saved to encoder_inference_snapshot.pickle")
