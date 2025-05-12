from fid.trainer.dataset import FidTestDataset
from fid.trainer.fid_trainer import FiDLightningModule
from fid.model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM, Qwen2FidDecoderConfig
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from torch.utils.data import DataLoader
import torch
import os


encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype="auto")
encoder.gradient_checkpointing_enable()

config = Qwen2FidDecoderConfig.from_json_file("./fid/model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, torch_dtype="auto", device_map="auto")
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

# 1st model
# model_path = r"/work/hdd/becw/sma2/cs598lmz/wandb-logs/lightning_logs/ahkjym1q/checkpoints/epoch=2-step=4335.ckpt"

# copy model
# model_path = r"/work/nvme/becw/sma2/cs598lmz/wandb-logs/lightning_logs/1vz75kyi/checkpoints/epoch=2-step=4335.ckpt"

# 8layer model
# model_path = r"/work/nvme/becw/sma2/cs598lmz/wandb-logs/lightning_logs/7vhu6kzi/checkpoints/epoch=1-step=2890.ckpt"

# full copy model
# model_path = r"/work/nvme/becw/sma2/cs598lmz/wandb-logs/lightning_logs/kzmrg4n9/checkpoints/epoch=2-step=4335.ckpt"

# full copy model batch size 512
model_path = r"/work/nvme/becw/sma2/cs598lmz/wandb-logs/lightning_logs/yu821i9m/checkpoints/epoch=9-step=3628.ckpt"

module = FiDLightningModule.load_from_checkpoint(model_path,
    encoder=encoder,
    decoder=decoder,
)

decoder = module.decoder
decoder.eval()
encoder = module.encoder
encoder.eval()


sample_context = """
# fid/trainer/fid_trainer.py
def fid_generate(encoder: Qwen2Model, decoder: Qwen2FidDecoderForCausalLM,
                 tokenizer: Qwen2TokenizerFast, context_text: list[str],
                 unfinished_code_text: str, max_encoder_tokens=24576,
                 max_decoder_tokens=8192, max_new_tokens: int = 10) -> str:
    
    context_inputs = tokenizer(context_text, return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=max_encoder_tokens).to(encoder.device)
    # Parallel encoding of context text
    encoder_hidden_states = encoder(**context_inputs).last_hidden_state
    print(f"encoder hidden states shape: {encoder_hidden_states.shape}")
    encoder_attention_mask = context_inputs['attention_mask']
    context_lengths = torch.sum(encoder_attention_mask, dim=1)

    # Trim padding and concatenate encoder hidden states
    trimmed_hidden_states = [hidden_state[:length] for hidden_state, length in zip(encoder_hidden_states, context_lengths)]
    encoder_hidden_states = torch.cat(trimmed_hidden_states, dim=0).unsqueeze(0)
    print(f"concat encoder hidden states shape: {encoder_hidden_states.shape}")

    unfinished_code_inputs = tokenizer(unfinished_code_text, return_tensors="pt",
                                       truncation=True, max_length=max_decoder_tokens).to(decoder.device)
    print(f"unfinished code inputs shape: {unfinished_code_inputs['input_ids'].shape}")

    # Run decoder generation
    outputs = decoder.generate(
        input_ids=unfinished_code_inputs['input_ids'],
        encoder_hidden_states=encoder_hidden_states,
        max_new_tokens=max_new_tokens,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text
"""


sample_code_to_complete = """
# fid/trainer/fid_trainer.py
def fid_generate(encoder: Qwen2Model, decoder: Qwen2FidDecoderForCausalLM,
                 tokenizer: Qwen2TokenizerFast, context_text: list[str],
                 unfinished_code_text: str, max_encoder_tokens=24576,
                 max_decoder_tokens=8192, max_new_tokens: int = 10) -> str:
    
    context_inputs = tokenizer(context_text, return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=max_encoder_tokens).to(encoder.device)
    # Parallel encoding of context text
    encoder_hidden_states = encoder(**context_inputs).last_hidden_state
    print(f"encoder hidden states shape: {encoder_hidden_states.shape}")
    encoder_attention_mask = context_inputs['attention_mask']
    context_lengths = torch.sum(encoder_attention_mask, dim=1)

    # Trim padding and concatenate encoder hidden states"""




def fid_generate(encoder: Qwen2Model, decoder: Qwen2FidDecoderForCausalLM,
                 tokenizer: Qwen2TokenizerFast, context_text: list[str],
                 unfinished_code_text: str, max_encoder_tokens=24576,
                 max_decoder_tokens=8192, max_new_tokens: int = 32) -> str:
    
    context_inputs = tokenizer(context_text, return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=max_encoder_tokens).to(encoder.device)
    # Parallel encoding of context text
    encoder_hidden_states = encoder(**context_inputs).last_hidden_state
    print(f"encoder hidden states shape: {encoder_hidden_states.shape}")
    encoder_attention_mask = context_inputs['attention_mask']
    context_lengths = torch.sum(encoder_attention_mask, dim=1)

    # Trim padding and concatenate encoder hidden states
    trimmed_hidden_states = [hidden_state[:length] for hidden_state, length in zip(encoder_hidden_states, context_lengths)]
    encoder_hidden_states = torch.cat(trimmed_hidden_states, dim=0).unsqueeze(0)
    print(f"concat encoder hidden states shape: {encoder_hidden_states.shape}")

    unfinished_code_inputs = tokenizer(unfinished_code_text, return_tensors="pt",
                                       truncation=True, max_length=max_decoder_tokens).to(decoder.device)
    print(f"unfinished code inputs shape: {unfinished_code_inputs['input_ids'].shape}")

    # Run decoder generation
    outputs = decoder.generate(
        input_ids=unfinished_code_inputs['input_ids'],
        attention_mask=unfinished_code_inputs['attention_mask'],
        encoder_hidden_states=encoder_hidden_states,
        max_new_tokens=max_new_tokens,
        eos_token_id=None
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return output_text


with torch.no_grad():
    result = fid_generate(encoder, decoder, tokenizer, [sample_context], sample_code_to_complete, max_new_tokens=128)
print(result)