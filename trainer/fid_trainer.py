import lightning as L
from model.modular_qwen2_fid import Qwen2FidDecoderForCausalLM
from transformers.models.qwen2 import Qwen2TokenizerFast, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim import AdamW
import torch
from torch import nn

def fid_batch_forward(encoder: Qwen2Model,
                      decoder: Qwen2FidDecoderForCausalLM,
                      batch: tuple[list[list[list[int]]], list[list[int]]],
                      pad_token_id: int,
                      device: str)-> CausalLMOutputWithPast:
    import pdb
    pdb.set_trace()
    
    encoder_input_ids, decoder_input_ids = batch
    encoder_context_num = [len(context_list) for context_list in encoder_input_ids]
    encoder_context_lengths = [
        [len(context) for context in context_list]
        for context_list in encoder_input_ids
    ]
    
    # Pad tokens and convert to tensors
    encoder_input_ids = [
        torch.Tensor(context, dtype=torch.long)
        for context_list in encoder_input_ids
        for context in context_list
    ]
    encoder_input_ids = nn.utils.rnn.pad_sequence(
        encoder_input_ids, batch_first=True, padding_value=pad_token_id,
        padding_side="right"
    )
    encoder_attention_mask = encoder_input_ids.ne(pad_token_id).long()
    decoder_input_ids = [
        torch.Tensor(context, dtype=torch.long)
        for context in decoder_input_ids
    ]
    decoder_input_ids = nn.utils.rnn.pad_sequence(
        decoder_input_ids, batch_first=True, padding_value=pad_token_id,
        padding_side="left"
    )
    decoder_attention_mask = decoder_input_ids.ne(pad_token_id).long()

    # Encoder forward
    encoder_input_ids = encoder_input_ids.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)
    encoder_hidden_states = encoder(
        encoder_input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask
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

    # Decoder forward
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)
    decoder_outputs = decoder.forward(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=cross_attn_mask,
        labels=decoder_input_ids,
        use_cache=False,
    )
    return decoder_outputs


class FiDLightningModule(L.LightningModule):
    def __init__(self,
                 encoder: Qwen2Model,
                 decoder: Qwen2FidDecoderForCausalLM,
                 tokenizer: Qwen2TokenizerFast,
                 lr: float = 1e-4):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = tokenizer.pad_token_id  # we only need the pad token from the tokenizer
        self.lr = lr

        # Freeze decoder parameters except for cross attention
        for name, param in self.decoder.named_parameters():
            if "cross_attn" in name or "cross_attn_layernorm" in name:
                param.requires_grad = True  # Keep cross-attention trainable
            else:
                param.requires_grad = False  # Freeze everything else
    

    def training_step(self,
                      batch: tuple[list[list[list[int]]], list[list[int]]],
                      batch_idx: int):
        outputs = fid_batch_forward(
            self.encoder,
            self.decoder,
            batch,
            self.pad_token_id,
            self.device
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        outputs = fid_batch_forward(
            self.encoder,
            self.decoder,
            batch,
            self.pad_token_id,
            self.device
        )
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True)


    def configure_optimizers(self):
        trainable_params = [
            p for p in self.decoder.parameters() if p.requires_grad
        ]
        trainable_params += [
            p for p in self.encoder.parameters()
        ]
        optimizer = AdamW(trainable_params, lr=self.lr)
        return optimizer
    
