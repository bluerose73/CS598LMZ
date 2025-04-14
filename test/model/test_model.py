from model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2TokenizerFast
from transformers import AutoModelForCausalLM


qwen2model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B")
print("loaded qwen2 model")

config = Qwen2FidDecoderConfig.from_pretrained("./model/config.json")
print(config)

fidmodel = Qwen2FidDecoderForCausalLM(config)

print(fidmodel.device)



missing_keys, unexpected_keys = fidmodel.load_state_dict(qwen2model.state_dict(), strict=False)
print("loaded state dict")
print(missing_keys)
print(unexpected_keys)


exit(0)

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-3B")

input_text = "import numpy as"
inputs = tokenizer(input_text, return_tensors="pt").to(fidmodel.device)

fid_output = fidmodel.forward(**inputs, use_cache=True)
qwen2_output = qwen2model.forward(**inputs, use_cache=True)

