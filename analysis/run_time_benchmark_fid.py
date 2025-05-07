#!/usr/bin/env python
import argparse
import time
import torch
import numpy as np
from transformers.models.qwen2 import Qwen2Model, Qwen2TokenizerFast, Qwen2ForCausalLM
from fid.model.configuration_qwen2_fid import Qwen2FidDecoderConfig
from fid.model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM


def run_once(encoder, model, device, vocab_size, args):
        # Create random prompts for encoder and decoder
        encoder_input_ids = torch.randint(0, vocab_size, (args.n_encoder_chunks, args.n_encoder_tokens_per_chunk), device=device)
        decoder_input_ids = torch.randint(0, vocab_size, (1, args.n_decoder_tokens), device=device)

        # --- Measure encode the context time ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_encode = time.perf_counter()
        with torch.no_grad():
            encoder_outputs = encoder(input_ids=encoder_input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        encode_time = time.perf_counter() - start_encode

        # Concatenate encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state.reshape(1, -1, encoder_outputs.last_hidden_state.shape[-1])

        # --- Measure prompt prefill time ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start_prefill

        # --- Measure decoding time ---
        last_token = decoder_input_ids[:, -1:]
        past = outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_decode = time.perf_counter()
        for _ in range(args.n_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=last_token, past_key_values=past, encoder_hidden_states=encoder_hidden_states, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            logits = outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            last_token = next_token
            past = outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        decoding_time = time.perf_counter() - start_decode
        
        return encode_time, prefill_time, decoding_time

def main():
    parser = argparse.ArgumentParser(description="Measure prompt prefill vs decoding times for Qwen2Fid.")
    parser.add_argument("--n_cross_attn_layers", type=int, default=1, help="Number of cross-attention layers")
    parser.add_argument("--n_encoder_tokens_per_chunk", type=int, default=2048, help="Number of tokens per encoder chunk")
    parser.add_argument("--n_encoder_chunks", type=int, required=True, help="Number of encoder chunks")
    parser.add_argument("--n_decoder_tokens", type=int, default=2048, help="Number of decoder input tokens")
    parser.add_argument("--n_new_tokens", type=int, default=32, help="Number of new tokens to generate")
    parser.add_argument("--n_run", type=int, default=5, help="Number of runs to average")
    args = parser.parse_args()

    # Load the tokenizer and model
    encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", device_map="auto", torch_dtype="auto")
    encoder.eval()

    config = Qwen2FidDecoderConfig.from_json_file("./fid/model/config.json")
    config.num_cross_attn_layers = args.n_cross_attn_layers
    # print("loaded fid model config")
    model = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, device_map="auto", torch_dtype="auto")
    model.eval()

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    model.to(device)

    encode_times = []
    prefill_times = []
    decoding_times = []

    # Use model's vocab_size from configuration
    vocab_size = model.config.vocab_size

    # Warm up the model by running it once
    run_once(encoder, model, device, vocab_size, args)

    # Run the measurement n_run times
    for run in range(args.n_run):
        encode_time, prefill_time, decoding_time = run_once(encoder, model, device, vocab_size, args)
        encode_times.append(encode_time)
        prefill_times.append(prefill_time)
        decoding_times.append(decoding_time)

    # Calculate and print the mean and standard deviation of the timings.
    encode_mean = np.mean(encode_times)
    encode_std = np.std(encode_times)
    prefill_mean = np.mean(prefill_times)
    prefill_std = np.std(prefill_times)
    decoding_mean = np.mean(decoding_times)
    decoding_std = np.std(decoding_times)

    print("--- Results ---")
    print(f"Model: Qwen2Fid | Cross attn layers: {args.n_cross_attn_layers} | Input tokens: {args.n_encoder_chunks * args.n_encoder_tokens_per_chunk + args.n_decoder_tokens} | New tokens: {args.n_new_tokens} | Runs: {args.n_run}")
    print(f"Encode context: mean time = {encode_mean:.4f} sec, std = {encode_std:.4f} sec")
    print(f"Prompt prefill: mean time = {prefill_mean:.4f} sec, std = {prefill_std:.4f} sec")
    print(f"Decoding: mean time = {decoding_mean:.4f} sec, std = {decoding_std:.4f} sec")
    print(f"Percent Prefill: {prefill_mean / (prefill_mean + decoding_mean) * 100:.2f}%")

if __name__ == "__main__":
    main()
