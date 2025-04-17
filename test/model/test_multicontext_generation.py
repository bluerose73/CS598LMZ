from model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM, Qwen2FidDecoderConfig
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2TokenizerFast
import torch

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

if __name__ == "__main__":

    encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
    encoder.eval()

    config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
    print("loaded fid model config")
    fidmodel = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
    fidmodel.eval()

    tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    print(f"loaded fid model to device {fidmodel.device}")
    print(f"fid model dtype: {fidmodel.dtype}")


    context_text = [
        "import numpy as npy\nimport pandas as pd\nimport matplotlib.pyplot as plt\n",
        "# This document is a test document",
    ]
    unfinished_code_text = "import numpy as"

    completion = fid_generate(encoder, fidmodel, tokenizer, context_text, unfinished_code_text)

    print(f"completion: {completion}")