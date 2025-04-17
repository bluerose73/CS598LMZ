from model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM, Qwen2FidDecoderConfig
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2TokenizerFast


encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
encoder.eval()
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
print("loaded fid model config")
print(f"attention implementation: {config._attn_implementation}")
fidmodel = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
print(f"attention implementation in model: {fidmodel.config._attn_implementation}")
print(f"attention autoset in model: {fidmodel.config._attn_implementation_autoset}")
fidmodel.eval()
print(f"loaded fid model to device {fidmodel.device}")
print(f"fid model dtype: {fidmodel.dtype}")


context_text = "import numpy as np\nimport matplotlib.pyplot as plt\n"
unfinished_code_text = "import pandas as"

context_inputs = tokenizer(context_text, return_tensors="pt").to(fidmodel.device)
unfinished_code_inputs = tokenizer(unfinished_code_text, return_tensors="pt").to(fidmodel.device)

encoder_hidden_states = encoder(**context_inputs).last_hidden_state
fid_output = fidmodel.forward(
    encoder_hidden_states=encoder_hidden_states,
    encoder_attention_mask=context_inputs["attention_mask"],
    **unfinished_code_inputs,
    use_cache=True,
)
print(f"fid logits shape: {fid_output.logits.shape}")
