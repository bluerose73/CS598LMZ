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