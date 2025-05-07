from ..model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model, Qwen2ForCausalLM
import torch
from torch import nn

class FidTestDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[tuple[list[list[int]], list[int], str]]) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], list[str]]:
        """
        Returns:
        - encoder_input_ids: torch.Tensor - Padded encoder input IDs.
        - encoder_attention_mask: torch.Tensor - Attention mask for the encoder.
        - decoder_input_ids: torch.Tensor - Padded decoder input IDs.
        - decoder_attention_mask: torch.Tensor - Attention mask for the decoder.
        - encoder_context_num: list[int] - Number of contexts for each sample.
        - encoder_context_lengths: list[list[int]] - Lengths of each context for each sample without padding.
        """
        # import pdb
        # pdb.set_trace()
        encoder_input_ids, decoder_input_ids, ground_truth_list = zip(*batch)

        encoder_context_num = [len(context_list) for context_list in encoder_input_ids]
        encoder_context_lengths = [
            len(context)
            for context_list in encoder_input_ids
            for context in context_list
        ]
        
        # Pad tokens and convert to tensors
        encoder_input_ids = [
            torch.tensor(context, dtype=torch.long)
            for context_list in encoder_input_ids
            for context in context_list
        ]
        encoder_input_ids = nn.utils.rnn.pad_sequence(
            encoder_input_ids, batch_first=True, padding_value=self.pad_token_id,
            padding_side="right"
        )
        encoder_attention_mask = encoder_input_ids.ne(self.pad_token_id).long()
        decoder_input_ids = [
            torch.tensor(context, dtype=torch.long)
            for context in decoder_input_ids
        ]
        decoder_input_ids = nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=self.pad_token_id,
            padding_side="left"
        )
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).long()
        
        return (
            encoder_input_ids,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_context_num,
            encoder_context_lengths,
            ground_truth_list
        )


class DecoderTestDataCollator:
    def __init__(self, pad_token_id: int, cross_file_context=True):
        self.pad_token_id = pad_token_id
        self.cross_file_context = cross_file_context

    def __call__(self, batch: list[tuple[list[list[int]], list[int], str]]) -> tuple[
            torch.Tensor, torch.Tensor, list[str]]:
        """
        Returns:
        - decoder_input_ids: torch.Tensor - Padded decoder input IDs.
        - decoder_attention_mask: torch.Tensor - Attention mask for the decoder.
        - ground_truth_list: list[str] - List of ground truth strings.
        """
        # import pdb
        # pdb.set_trace()
        encoder_input_ids, decoder_input_ids, ground_truth_list = zip(*batch)


        full_decoder_input_ids = []
        for context_list, decoder_inputs in zip(encoder_input_ids, decoder_input_ids):
            full_decoder_input = []
            if self.cross_file_context:
                for context in context_list:
                    full_decoder_input.extend(context)
            full_decoder_input.extend(decoder_inputs)
            full_decoder_input_ids.append(full_decoder_input)

        decoder_input_ids = [
            torch.tensor(context, dtype=torch.long)
            for context in full_decoder_input_ids
        ]
        decoder_input_ids = nn.utils.rnn.pad_sequence(
            decoder_input_ids, batch_first=True, padding_value=self.pad_token_id,
            padding_side="left"
        )
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).long()
        
        return (
            decoder_input_ids,
            decoder_attention_mask,
            ground_truth_list
        )


def fid_batch_generate(encoder: Qwen2Model,
                      decoder: Qwen2FidDecoderForCausalLM,
                      tokenizer: Qwen2TokenizerFast,
                      batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[int]]],
                      device: str)-> list[dict]:
    
    (encoder_input_ids, encoder_attention_mask,
     decoder_input_ids, decoder_attention_mask,
     encoder_context_num, encoder_context_lengths,
     grond_truth_list) = batch

    # Encoder forward
    encoder_input_ids = encoder_input_ids.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)
    encoder_hidden_states = encoder(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        use_cache=False,
    ).last_hidden_state

    # Concat encoder hidden states
    encoder_hidden_states_by_sample = []
    context_id_start = 0
    for context_num in encoder_context_num:
        context_id_end = context_id_start + context_num
        encoder_hidden_states_split = [
            encoder_hidden_states[i, :context_length]
            for i, context_length in enumerate(encoder_context_lengths[context_id_start:context_id_end])
        ]
        encoder_hidden_states_by_sample.append(
            torch.cat(encoder_hidden_states_split, dim=0)
        )
        context_id_start = context_id_end
    encoder_hidden_states = nn.utils.rnn.pad_sequence(
        encoder_hidden_states_by_sample, batch_first=True, padding_value=0,
        padding_side="right"
    )
    cross_attn_mask = torch.zeros(
        encoder_hidden_states.size()[:2],  # shape: (batch_size, seq_len)
        dtype=torch.long,
        device=encoder_hidden_states.device,
    ).to(device)
    for i, seq in enumerate(encoder_hidden_states_by_sample):
        cross_attn_mask[i, :seq.size(0)] = 1

    # Decoder generation
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)
    decoder_outputs = decoder.generate(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=cross_attn_mask,
        max_new_tokens=32,
    )
    # Now split prompt and completion
    output_texts = []
    for generated_ids, prompt_ids in zip(decoder_outputs, decoder_input_ids):
        prompt_len = prompt_ids.shape[0]
        generated_prompt = tokenizer.decode(generated_ids[:prompt_len], skip_special_tokens=True)
        generated_completion = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)
        output_texts.append((generated_prompt, generated_completion))

    # Return structure
    return [
        {
            "prompt": prompt_text,
            "completion": completion_text,
            "ground_truth": ground_truth,
        }
        for (prompt_text, completion_text), ground_truth in zip(output_texts, grond_truth_list)
    ]


def qwen2_batch_generate(decoder: Qwen2ForCausalLM,
                      tokenizer: Qwen2TokenizerFast,
                      batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[int]]],
                      device: str)-> list[dict]:
    
    decoder_input_ids, decoder_attention_mask, ground_truth_list = batch

    # Decoder generation
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)
    decoder_outputs = decoder.generate(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        max_new_tokens=32,
    )
    # Now split prompt and completion
    output_texts = []
    for generated_ids, prompt_ids in zip(decoder_outputs, decoder_input_ids):
        prompt_len = prompt_ids.shape[0]
        generated_prompt = tokenizer.decode(generated_ids[:prompt_len], skip_special_tokens=True)
        generated_completion = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)
        output_texts.append((generated_prompt, generated_completion))

    # Return structure
    return [
        {
            "prompt": prompt_text,
            "completion": completion_text,
            "ground_truth": ground_truth,
        }
        for (prompt_text, completion_text), ground_truth in zip(output_texts, ground_truth_list)
    ]
