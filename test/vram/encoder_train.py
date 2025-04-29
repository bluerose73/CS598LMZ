# How much GPU memory does the encoder-decoder model use?

import lightning as L
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
import torch
from torch.optim import AdamW
from torch.cuda import OutOfMemoryError


# TF32 tensor cores
torch.set_float32_matmul_precision("high")

torch.cuda.memory._record_memory_history()

# Load models
encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
encoder.gradient_checkpointing_enable()
encoder.train()
config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
# decoder.gradient_checkpointing_enable()
decoder.train()
for name, param in decoder.named_parameters():
    if "cross_attn" in name or "cross_attn_layernorm" in name:
        param.requires_grad = True  # Keep cross-attention trainable
    else:
        param.requires_grad = False  # Freeze everything else


tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

print(f"peak memory allocated for models: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

optimizer = AdamW(list(encoder.parameters()) + [param for param in decoder.parameters() if param.requires_grad], lr=1e-5)

print(f"peak memory allocated after optimizer creation: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")  


# Create dummy input
input_length = 2048
n_context = 128
dummy_input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(n_context, input_length),
    dtype=torch.long,
    device=encoder.device,
)
decoder_input_length = 2048
dummy_decoder_input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(16, decoder_input_length),
    dtype=torch.long,
    device=encoder.device,
)


try:
    # Run encoder forward pass with grad
    encoder_outputs = encoder(input_ids=dummy_input_ids)

    print(f"peak memory allocated after encoder forward: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    hidden_states = encoder_outputs.last_hidden_state

    print(f"hidden_states shape: {hidden_states.shape}")
    # concate along batch dimension
    hidden_states = hidden_states.view(16, -1, hidden_states.shape[-1])
    # hidden_states = hidden_states.unsqueeze(0)
    print(f"hidden_states shape after unsqueeze: {hidden_states.shape}")

    print(f"hidden_states.requires_grad: {hidden_states.requires_grad}")
    print(f"dummpy_decoder_input_ids.requires_grad: {dummy_decoder_input_ids.requires_grad}")

    # Run decoder forward pass with grad
    decoder_outputs = decoder(
        input_ids=dummy_decoder_input_ids,
        encoder_hidden_states=hidden_states,
        use_cache=False,
        labels=dummy_decoder_input_ids,
    )

    print(f"peak memory allocated after decoder forward: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    loss = decoder_outputs.loss

    optimizer.zero_grad()

    loss.backward()
    print(f"peak memory allocated after backward: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    optimizer.step()
    print(f"peak memory allocated after optimizer step: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

except OutOfMemoryError as e:
    print(f"Out of memory error: {e}")

# Save CUDA memory snapshot
torch.cuda.memory._dump_snapshot("encoder_train_snapshot.pickle")
print("CUDA memory snapshot saved to encoder_train_snapshot.pickle")
