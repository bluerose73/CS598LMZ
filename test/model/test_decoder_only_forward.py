from model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2TokenizerFast, Qwen2Config
import torch
import types
from my_logger import my_logger

def print_param_names_and_dtypes(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")


qwen2model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", device_map="auto", torch_dtype="auto")
print(f"loaded qwen2 model to device {qwen2model.device}")
print(f"qwen2 model dtype: {qwen2model.dtype}")


config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
print("loaded fid model config")

# fidmodel = Qwen2FidDecoderForCausalLM(config)
# fidmodel.to(qwen2model.dtype)
# fidmodel.to(qwen2model.device)
fidmodel = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")

qwen2model.eval()
fidmodel.eval()

print_param_names_and_dtypes(fidmodel)

# missing_keys, unexpected_keys = fidmodel.load_state_dict(qwen2model.state_dict(), strict=False)
# print("loaded state dict")
# print(f"missing keys: {missing_keys}")
# print(f"unexpected keys: {unexpected_keys}")

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")

input_text = "import numpy as"
inputs = tokenizer(input_text, return_tensors="pt").to(fidmodel.device)


original_update_causal_mask = qwen2model.model._update_causal_mask

def update_causal_mask_w_print(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    output_attentions: bool,
):
    # Call the original method instead of self._update_causal_mask
    causal_mask = original_update_causal_mask(
        attention_mask=attention_mask,
        input_tensor=input_tensor,
        cache_position=cache_position,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
    )
    if causal_mask is not None:
        print(f"causal_mask shape: {causal_mask.shape}")
    else:
        print("causal_mask is None")
    return causal_mask

# Override the method with the new wrapped version
qwen2model.model._update_causal_mask = types.MethodType(update_causal_mask_w_print, qwen2model.model)

qwen2_output = qwen2model.forward(**inputs, use_cache=True, output_hidden_states=True)
fid_output = fidmodel.forward(**inputs, use_cache=True, output_hidden_states=True)

print(f"qwen2 logits shape: {qwen2_output.logits.shape}")
print(f"fid logits shape: {fid_output.logits.shape}")

if torch.equal(fid_output.logits[0, 0], qwen2_output.logits[0, 0]):
    print("The logits for the first token are the same")
else:
    print("The logits for the first token are different")
    

if torch.equal(fid_output.logits, qwen2_output.logits):
    print("The logits are the same")
else:
    print("The logits are different")
    print(f"fidmodel logits[0, 1]: {fid_output.logits[0, 1, :10]}")
    print(f"qwen2model logits[0, 1]: {qwen2_output.logits[0, 1, :10]}")
    print(f"fidmodel logits[0, 1] max: {fid_output.logits[0, 1].max()}")
    print(f"qwen2model logits[0, 1] max: {qwen2_output.logits[0, 1].max()}")

exit(0)

qwen2_kv_cache = qwen2_output.past_key_values
fid_kv_cache = fid_output.past_key_values[0]

print(f"qwen2 kv_cache type: {type(qwen2_kv_cache)}")
print(f"fid kv_cache type: {type(fid_output.past_key_values)}")
print(f"fid kv_cache[0] type: {type(fid_kv_cache)}")

qwen2_key_cache = qwen2_kv_cache.key_cache[35]
fid_key_cache = fid_kv_cache.key_cache[35]

print(f"qwen2 key_cache[35] shape: {qwen2_key_cache.shape}")
print(f"fid key_cache[35] shape: {fid_key_cache.shape}")


if torch.equal(fid_key_cache, qwen2_key_cache):
    print("The key_cache is the same")
else:
    print("The key_cache is different")


print(f"fid key_cache values: {fid_key_cache[0, :, 2, :10]}")
print(f"qwen2 key_cache values: {qwen2_key_cache[0, :, 2, :10]}")


qwen2_hidden = qwen2_output.hidden_states
# for i, hidden in enumerate(qwen2_hidden):
#     print(f"qwen2 hidden state {i} shape: {hidden.shape}")
fid_hidden = fid_output.hidden_states
# for i, hidden in enumerate(fid_hidden):
#     print(f"fid hidden state {i} shape: {hidden.shape}")

print(f"qwen2 hidden type: {type(qwen2_hidden)}")
print(f"qwen2 hidden[0] shape: {qwen2_hidden[0].shape}")


if torch.equal(qwen2_hidden[34], fid_hidden[34]):
    print("The hidden states 34 are the same")
else:
    print("The hidden states 34 are different")
    
if torch.equal(qwen2_hidden[35], fid_hidden[35]):
    print("The hidden states 35 are the same")
else:
    print("The hidden states 35 are different")

if torch.equal(qwen2_hidden[36], fid_hidden[36]):
    print("The hidden states 36 are the same")
else:
    print("The hidden states 36 are different")

for token in range(3):
    for layer in range(37):
        if torch.equal(qwen2_hidden[layer][0, token], fid_hidden[layer][0, token]):
            print(f"hidden states {layer} token {token} equal: True")
        else:
            print(f"hidden states {layer} token {token} equal: False")
            


# print(f"qwen2 key_cache[35] type: {type(qwen2_kv_cache.key_cache[35])}")
# print(f"fid key_cache[35] type: {type(fid_kv_cache.key_cache[35])}")
# print(f"qwen2 key_cache[35] length: {len(qwen2_kv_cache.key_cache[35])}")
# print(f"fid key_cache[35] length: {len(fid_kv_cache.key_cache[35])}")
# print(f"qwen2 key_cache[35][0] shape: {qwen2_kv_cache.key_cache[35][0].shape}")
# print(f"fid key_cache[35][0] shape: {fid_kv_cache.key_cache[35][0].shape}")


print("\n=== my logger ===")
qwen2_hidden_states = my_logger["hidden_states"][0]
fid_hidden_states = my_logger["hidden_states"][1]
if torch.equal(qwen2_hidden_states, fid_hidden_states):
    print("The hidden states are the same")
else:
    print("The hidden states are different")


qwen2_key_states = my_logger["key_states"][0]
fid_key_states = my_logger["key_states"][1]

if torch.equal(qwen2_key_states, fid_key_states):
    print("The key states are the same")
else:
    print("The key states are different")


qwen2_position_embedding_sin = my_logger["position_embeddings"][0][0]
fid_position_embedding_sin = my_logger["position_embeddings"][1][0]
if torch.equal(qwen2_position_embedding_sin, fid_position_embedding_sin):
    print("The positional embedding sin values are the same")
else:
    print("The positional embedding sin values are different")


qwen2_keyproj: torch.nn.Linear = qwen2model.model.layers[0].self_attn.k_proj
qwen2weight = qwen2_keyproj.weight
qwen2bias = qwen2_keyproj.bias

fid_keyproj: torch.nn.Linear = fidmodel.model.layers[0].self_attn.k_proj
fidweight = fid_keyproj.weight
fidbias = fid_keyproj.bias

if torch.equal(qwen2weight, fidweight):
    print("The k_proj weights are the same")
else:
    print("The k_proj weights are different")

if torch.equal(qwen2bias, fidbias):
    print("The k_proj biases are the same")
else:
    print("The k_proj biases are different")



qwen2_position_ids = my_logger["position_ids"][0]
fid_position_ids = my_logger["position_ids"][1]
if torch.equal(qwen2_position_ids, fid_position_ids):
    print("The position ids are the same")
else:
    print("The position ids are different")

qwen2_model_position_embeddings = my_logger["model_position_embeddings"][0]
fid_model_position_embeddings = my_logger["model_position_embeddings"][1]
if torch.equal(qwen2_model_position_embeddings, fid_model_position_embeddings):
    print("The model position embeddings are the same")
else:
    print("The model position embeddings are different")

qwen2_model_inputs_embeds = my_logger["inputs_embeds"][0]
fid_model_inputs_embeds = my_logger["inputs_embeds"][1]
if torch.equal(qwen2_model_inputs_embeds, fid_model_inputs_embeds):
    print("The model inputs embeddings are the same")
else:
    print("The model inputs embeddings are different")


qwen2_cos = my_logger["cos"][0]
fid_cos = my_logger["cos"][1]
if torch.equal(qwen2_cos, fid_cos):
    print("The cos values are the same")
else:
    print("The cos values are different")

qwen2_sin = my_logger["sin"][0]
fid_sin = my_logger["sin"][1]
if torch.equal(qwen2_sin, fid_sin):
    print("The sin values are the same")
else:
    print("The sin values are different")

qwen2_inv_freq = my_logger["inv_freq"][0]
fid_inv_freq = my_logger["inv_freq"][1]
if torch.equal(qwen2_inv_freq, fid_inv_freq):
    print("The inv_freq values are the same")
else:
    print("The inv_freq values are different")

qwen2_inv_freq_expanded = my_logger["inv_freq_expanded"][0]
fid_inv_freq_expanded = my_logger["inv_freq_expanded"][1]
if torch.equal(qwen2_inv_freq_expanded, fid_inv_freq_expanded):
    print("The inv_freq_expanded values are the same")
else:
    print("The inv_freq_expanded values are different")

qwen2_position_ids_expanded = my_logger["position_ids_expanded"][0]
fid_position_ids_expanded = my_logger["position_ids_expanded"][1]
if torch.equal(qwen2_position_ids_expanded, fid_position_ids_expanded):
    print("The position_ids_expanded values are the same")
else:
    print("The position_ids_expanded values are different")

qwen2_freqs = my_logger["freqs"][0]
fid_freqs = my_logger["freqs"][1]
if torch.equal(qwen2_freqs, fid_freqs):
    print("The freqs values are the same")
else:
    print("The freqs values are different")

qwen2_inv = my_logger["self.inv_freq_before_update"][0]
fid_inv = my_logger["self.inv_freq_before_update"][1]
if torch.equal(qwen2_inv, fid_inv):
    print("The inv_freq_before_update values are the same")
else:
    print("The inv_freq_before_update values are different")

qwen2_inv = my_logger["self.inv_freq_after_update"][0]
fid_inv = my_logger["self.inv_freq_after_update"][1]
if torch.equal(qwen2_inv, fid_inv):
    print("The inv_freq_after_update values are the same")
else:
    print("The inv_freq_after_update values are different")

print(f"logger size: {len(my_logger["freqs"])}")