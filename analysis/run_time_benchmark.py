#!/usr/bin/env python
import argparse
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_once(model, device, vocab_size, args):
        # Create a random prompt of shape [1, n_input_tokens]
        input_ids = torch.randint(0, vocab_size, (1, args.n_input_tokens), device=device)

        # --- Measure prompt prefill time ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        with torch.no_grad():
            # Run the prompt through the model to get the past_key_values cache.
            outputs = model(input_ids=input_ids, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start_prefill

        # --- Measure decoding time ---
        # Begin with the last token of the prompt.
        last_token = input_ids[:, -1:]
        past = outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_decode = time.perf_counter()
        # Generate tokens one-by-one.
        for _ in range(args.n_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=last_token, past_key_values=past, use_cache=True)
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
    parser = argparse.ArgumentParser(description="Measure prompt prefill vs decoding times for a Causal LM.")
    parser.add_argument("--model", type=str, help="Model name or path", default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--n_input_tokens", type=int, help="Number of input tokens")
    parser.add_argument("--n_new_tokens", type=int, help="Number of new tokens to generate", default=32)
    parser.add_argument("--n_run", type=int, help="Number of runs to average", default=5)
    args = parser.parse_args()

    # Load the tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
    model.eval()

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prefill_times = []
    decoding_times = []

    # Use model's vocab_size from configuration
    vocab_size = model.config.vocab_size

    # Warm up the model by running it once
    run_once(model, device, vocab_size, args)

    # Run the measurement n_run times
    for run in range(args.n_run):
        prefill_time, decoding_time = run_once(model, device, vocab_size, args)
        prefill_times.append(prefill_time)
        decoding_times.append(decoding_time)

    # Calculate and print the mean and standard deviation of the timings.
    prefill_mean = np.mean(prefill_times)
    prefill_std = np.std(prefill_times)
    decoding_mean = np.mean(decoding_times)
    decoding_std = np.std(decoding_times)

    print("--- Results ---")
    print(f"Model: {args.model} | Input tokens: {args.n_input_tokens} | New tokens: {args.n_new_tokens} | Runs: {args.n_run}")
    print(f"Prompt prefill: mean time = {prefill_mean:.4f} sec, std = {prefill_std:.4f} sec")
    print(f"Decoding: mean time = {decoding_mean:.4f} sec, std = {decoding_std:.4f} sec")
    print(f"Percent Prefill: {prefill_mean / (prefill_mean + decoding_mean) * 100:.2f}%")

if __name__ == "__main__":
    main()
