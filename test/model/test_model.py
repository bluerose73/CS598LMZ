from fid.model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM


config = Qwen2FidDecoderConfig.from_json_file("./fid/model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, torch_dtype="auto")

for name, param in decoder.named_parameters():
    print(name, param.size())