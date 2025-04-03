#!/usr/bin/env python
import argparse
import time
import torch
import numpy as np
from transformers.models.qwen2 import Qwen2Model, Qwen2TokenizerFast, Qwen2ForCausalLM
from model.configuration_qwen2_fid import Qwen2FidDecoderConfig
from model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM


def run_once(encoder, model, device, vocab_size, args):
        encoder_n_input_tokens = args.n_input_tokens * 3 // 4
        decoder_n_input_tokens = args.n_input_tokens - encoder_n_input_tokens

        # Create a random prompt of shape [1, n_input_tokens]
        encoder_input_ids = torch.randint(0, vocab_size, (1, encoder_n_input_tokens), device=device)
        decoder_input_ids = torch.randint(0, vocab_size, (1, decoder_n_input_tokens), device=device)

        # --- Encode the context ---
        encoder_output = encoder(input_ids=encoder_input_ids)
        encoder_hidden_states = encoder_output.last_hidden_state

        # --- Measure prompt prefill time ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        with torch.no_grad():
            # Run the prompt through the model to get the past_key_values cache.
            outputs = model(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start_prefill


        # --- Measure decoding time ---
        # Begin with the last token of the prompt.
        last_token = decoder_input_ids[:, -1:]
        past = outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_decode = time.perf_counter()
        # Generate tokens one-by-one.
        for _ in range(args.n_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=last_token, past_key_values=past, encoder_hidden_states=encoder_hidden_states, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            logits = outputs.logits
            # Greedy decode: select the token with the highest logit
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            last_token = next_token
            past = outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        decoding_time = time.perf_counter() - start_decode
        
        return prefill_time, decoding_time


def main():
    parser = argparse.ArgumentParser(description="Measure prompt prefill vs decoding times for Qwen2Fid.")
    parser.add_argument("--n_cross_attn_layers", type=int, required=True, help="Number of cross-attention layers")
    parser.add_argument("--n_input_tokens", type=int, required=True, help="Number of input tokens")
    parser.add_argument("--n_new_tokens", type=int, required=True, help="Number of new tokens to generate")
    parser.add_argument("--n_run", type=int, required=True, help="Number of runs to average")
    args = parser.parse_args()

    # Load the tokenizer and model
    encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
    encoder.eval()

    config = Qwen2FidDecoderConfig.from_json_file("./model/config.json")
    config.num_cross_attn_layers = args.n_cross_attn_layers
    print("loaded fid model config")
    model = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
    model.eval()

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    model.to(device)

    prefill_times = []
    decoding_times = []

    # Use model's vocab_size from configuration
    vocab_size = model.config.vocab_size

    # Warm up the model by running it once
    run_once(encoder, model, device, vocab_size, args)

    # Run the measurement n_run times
    for run in range(args.n_run):
        prefill_time, decoding_time = run_once(encoder, model, device, vocab_size, args)
        prefill_times.append(prefill_time)
        decoding_times.append(decoding_time)

    # Calculate and print the mean and standard deviation of the timings.
    prefill_mean = np.mean(prefill_times)
    prefill_std = np.std(prefill_times)
    decoding_mean = np.mean(decoding_times)
    decoding_std = np.std(decoding_times)

    print("--- Results ---")
    print(f"Model: Qwen2Fid | Cross attn layers: {args.n_cross_attn_layers} | Input tokens: {args.n_input_tokens} | New tokens: {args.n_new_tokens} | Runs: {args.n_run}")
    print(f"Prompt prefill: mean time = {prefill_mean:.4f} sec, std = {prefill_std:.4f} sec")
    print(f"Decoding: mean time = {decoding_mean:.4f} sec, std = {decoding_std:.4f} sec")
    print(f"Percent Prefill: {prefill_mean / (prefill_mean + decoding_mean) * 100:.2f}%")

if __name__ == "__main__":
    main()
