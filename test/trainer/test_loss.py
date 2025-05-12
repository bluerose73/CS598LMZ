from fid.trainer.fid_datamodule import FidTrainingDataModule
import lightning as L
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from fid.model.modular_qwen2_fid import Qwen2FidDecoderConfig, Qwen2FidDecoderForCausalLM
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from icecream import ic

L.seed_everything(42)

# TF32 tensor cores
torch.set_float32_matmul_precision("high")

encoder = Qwen2Model.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype="auto", device_map="auto")
encoder.gradient_checkpointing_enable()

config = Qwen2FidDecoderConfig.from_json_file("./fid/model/config.json")
decoder = Qwen2FidDecoderForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B", config=config, torch_dtype="auto", device_map="auto")
# decoder.gradient_checkpointing_enable()

tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
lr = 0.0003

ic(tokenizer.pad_token_id)
ic(tokenizer.eos_token_id)

per_device_batch_size = 4


ic(decoder.loss_function)
ic(decoder.loss_type)

# exit(0)

datamodule = FidTrainingDataModule(
    tokenized_data_load_dir="/work/nvme/becw/sma2/the-stack-v2-20k/tokenized",
    tokenizer=tokenizer,
    batch_size=per_device_batch_size,
    copy_ratio=1,
    post_process_max_context_num=1,
)


datamodule.setup("fit")
train_dataloader = datamodule.train_dataloader()


def fid_batch_forward_test(encoder: Qwen2Model,
                      decoder: Qwen2FidDecoderForCausalLM,
                      batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[list[int]]],
                      pad_token_id: int,
                      device: str)-> CausalLMOutputWithPast:
    
    (encoder_input_ids, encoder_attention_mask,
     decoder_input_ids, decoder_attention_mask,
     encoder_context_num, encoder_context_lengths) = batch


    ic(encoder_context_num)
    ic(encoder_context_lengths)

    ic(encoder_input_ids.size())
    ic(encoder_attention_mask.size())
    ic(decoder_input_ids.size())
    ic(decoder_attention_mask.size())

    # ic('encoder input for batch[3]:')
    # encoder_input_text = tokenizer.decode(encoder_input_ids[3].tolist(), skip_special_tokens=False)
    # print(encoder_input_text)


    # ic('decoder input for batch[3]:')
    # decoder_input_text = tokenizer.decode(decoder_input_ids[3].tolist(), skip_special_tokens=False)
    # print(decoder_input_text)

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

    ic(encoder_hidden_states.size())
    ic(cross_attn_mask.size())
    ic('non-zero count in cross_attn_mask:')
    ic(cross_attn_mask.count_nonzero(dim=-1))

    # Replace pad_token_id with -100 in labels
    labels = decoder_input_ids.clone()
    labels[labels == pad_token_id] = -100

    ic(labels.size())
    ic(decoder_input_ids[3])
    ic(labels[3])

    # test a single-batch forward
    encoder_hidden_states = encoder_hidden_states[3:4]
    cross_attn_mask = cross_attn_mask[3:4]
    input_ids = decoder_input_ids[3:4]
    attention_mask = decoder_attention_mask[3:4]
    labels = labels[3:4]
    ic('single batch forward')
    ic(encoder_hidden_states.size())
    ic(cross_attn_mask.size())
    ic(input_ids.size())
    ic(attention_mask.size())

    # Decoder forward
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    decoder_outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=cross_attn_mask,
        labels=labels,  # Use modified labels
        use_cache=False,
    )
    ic(decoder_outputs.loss)


    # trim to 400d
    encoder_hidden_states = encoder_hidden_states[:, :400, :]
    cross_attn_mask = cross_attn_mask[:, :400]
    input_ids = input_ids[:, :400]
    attention_mask = attention_mask[:, :400]
    labels = labels[:, :400]
    ic('trimmed single batch forward')
    decoder_outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=cross_attn_mask,
        labels=labels,  # Use modified labels
        use_cache=False,
    )
    ic(decoder_outputs.loss)

    return decoder_outputs




batch = next(iter(train_dataloader))

with torch.no_grad():
    
    fid_batch_forward_test(
        encoder=encoder,
        decoder=decoder,
        batch=batch,
        pad_token_id=tokenizer.pad_token_id,
        device="cuda:0"
    )