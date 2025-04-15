from .configuration_qwen2_fid import Qwen2FidDecoderConfig
from torch import nn
import torch
from typing import Tuple, Optional, Callable, Union, List, Dict
import warnings
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (eager_attention_forward,
                                                      Qwen2Attention,
                                                      Qwen2MLP,
                                                      Qwen2RMSNorm,
                                                      Qwen2PreTrainedModel,
                                                      Qwen2DecoderLayer,
                                                      Qwen2RotaryEmbedding
)
from transformers.utils.logging import get_logger
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation import GenerationMixin, GenerationConfig


logger = get_logger(__name__)


class Qwen2CrossAttention(nn.Module):
    """Multi-headed cross attention module for Qwen2.

    This module computes cross attention over encoder hidden states.
    The query is projected from the decoder hidden states, and the keys and values
    are projected from the encoder hidden states.
    """
    def __init__(self, config: Qwen2FidDecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # Cross attention is non-causal

        # Query projection: from decoder hidden states.
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        # Key and value projections: from encoder hidden states.
        input_hidden_size = config.encoder_hidden_size
        self.k_proj = nn.Linear(input_hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(input_hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch_size, q_len, hidden_size]
        encoder_hidden_states: torch.Tensor,  # [batch_size, k_len, encoder_hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug("====== Qwen2CrossAttention forward ====")
        logger.debug(f"attention_mask.shape = {attention_mask.shape if attention_mask is not None else None}")
        bsz, q_len, _ = hidden_states.size()
        bsz_enc, k_len, _ = encoder_hidden_states.size()
        assert bsz == bsz_enc, "Batch size mismatch between decoder and encoder."

        # Compute query states from decoder hidden states.
        query_states = self.q_proj(hidden_states)
        # Reshape to [batch_size, num_attention_heads, q_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)

        # Handle caching for key and value states using DynamicCache.
        if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx and \
           past_key_value.key_cache[self.layer_idx] != []:
            # Reuse cached key/value states if available.
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            # Compute key and value states from encoder hidden states.
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            key_states = key_states.view(bsz, k_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, k_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            # Update cache if provided.
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": None}
                )

        # Choose the attention interface (e.g. eager or optimized implementations)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Compute scaled dot-product attention.
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=False,
            **kwargs,
        )

        # Reshape and project output back to hidden size.
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Qwen2FidDecoderLayer(nn.Module):
    """
    Qwen2FidDecoderLayer implements a decoder layer with an extra cross attention
    block (Fusion-In Decoder). It first applies self-attention over decoder hidden states,
    then cross attention with encoder hidden states, followed by an MLP.
    
    Args:
        config (Qwen2FidDecoderConfig): The model configuration containing hyperparameters.
        layer_idx (int): Index of the current layer.
    """
    def __init__(self, config: Qwen2FidDecoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention from Qwen2 (causal attention)
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        # Cross-attention over encoder hidden states (non-causal)
        self.cross_attn = Qwen2CrossAttention(config=config, layer_idx=layer_idx)
        # Feed-forward network (MLP)
        self.mlp = Qwen2MLP(config)
        
        # LayerNorms before each major sub-layer.
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[Cache, Cache]] = None,  # self and cross attention caches
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        self_attn_cache, cross_attn_cache = past_key_value if past_key_value is not None else (None, None)

        logger.debug("====== Qwen2FidDecoderLayer forward ======")
        logger.debug(f"layer_idx {self.layer_idx}")
        logger.debug(f"hidden_states shape: {hidden_states.shape}")
        logger.debug(f"encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else None}")
        logger.debug(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        logger.debug(f"encoder_attention_mask shape: {encoder_attention_mask.shape if encoder_attention_mask is not None else None}")

        # ===== Self-Attention Block =====
        residual = hidden_states
        # Normalize input for self-attention.
        normed_hidden_states = self.input_layernorm(hidden_states)
        self_attn_output, self_attn_weights = self.self_attn(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=self_attn_cache,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self_attn_output

        # ===== Cross-Attention Block =====
        if encoder_hidden_states is not None:
            residual = hidden_states
            normed_hidden_states = self.cross_attn_layernorm(hidden_states)
            cross_attn_output, cross_attn_weights = self.cross_attn(
                hidden_states=normed_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_cache,
                **kwargs,
            )
            hidden_states = residual + cross_attn_output

        # ===== MLP (Feed-Forward) Block =====
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            # Return both self and cross attention weights.
            outputs += ((self_attn_weights, cross_attn_weights),)
        return outputs


class Qwen2FidDecoderPretrainedModel(Qwen2PreTrainedModel):
    config_class = Qwen2FidDecoderConfig
    _no_split_modules = ["Qwen2FidDecoderLayer", "Qwen2DecoderLayer"]



class Qwen2FidDecoderModel(Qwen2FidDecoderPretrainedModel):
    """
    Qwen2FidDecoderModel is a decoder-only transformer that integrates Fusion-In Decoder (FID)
    cross-attention layers into the top layers of the network. The first 
    (num_hidden_layers - num_cross_attn_layers) layers are standard Qwen2DecoderLayer modules,
    and the top num_cross_attn_layers layers are replaced by Qwen2FidDecoderLayer modules which 
    include an extra cross attention block over encoder hidden states.
    
    Args:
        config (Qwen2FidDecoderConfig): Model configuration containing hyperparameters,
            including num_hidden_layers, num_cross_attn_layers, encoder_hidden_size, etc.
    """
    def __init__(self, config: Qwen2FidDecoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Build the decoder layers: lower layers are standard; top layers use FID with cross attention.
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx < config.num_hidden_layers - config.num_cross_attn_layers:
                self.layers.append(Qwen2DecoderLayer(config, layer_idx))
            else:
                self.layers.append(Qwen2FidDecoderLayer(config, layer_idx))
                
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Cache, Cache]] = None,  # (self_attention_cache, cross_attention_cache)
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize caching as a tuple of caches if caching is enabled and no cache is provided.
        if use_cache and past_key_values is None:
            past_key_values = (DynamicCache(), DynamicCache())

        # Determine past seen tokens from the self-attention cache (first element of tuple).
        if cache_position is None:
            past_seen_tokens = past_key_values[0].get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values[0], output_attentions
        )

        logger.debug("====== Qwen2FidDecoderModel forward ======")
        if causal_mask is not None:
            logger.debug(f"causal_mask shape: {causal_mask.shape}")
        else:
            logger.debug("causal_mask is None")
        logger.debug(f"input_embeds shape: {inputs_embeds.shape}")

        if self.config._attn_implementation != "sdpa":
            raise NotImplementedError(f"encoder attention mask is only tested for sdpa, but current attention is {self.config._attn_implementation}")
        if encoder_attention_mask is not None:
            attention_mask_converter = AttentionMaskConverter(is_causal=False)
            encoder_attention_mask = attention_mask_converter.to_4d(
                encoder_attention_mask,
                query_length=inputs_embeds.shape[1],
                dtype=inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds

        # Create shared rotary position embeddings.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Loop through each decoder layer.
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Prepare the appropriate cache for the layer.
            # FID layers expect a tuple of caches; standard layers expect only the self-attention cache.
            if isinstance(layer, Qwen2FidDecoderLayer):
                layer_cache = past_key_values  # tuple: (self_attention_cache, cross_attention_cache)
            else:
                layer_cache = past_key_values[0]

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpointing is not tested for FID layers.")
                if isinstance(layer, Qwen2FidDecoderLayer):
                    def custom_forward(*inputs):
                        return layer(
                            *inputs,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            **flash_attn_kwargs,
                        )
                    layer_outputs = self._gradient_checkpointing_func(
                        custom_forward,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        layer_cache,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        layer_cache,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
            else:
                if isinstance(layer, Qwen2FidDecoderLayer):
                    # For FID layers, pass the tuple of caches along with encoder inputs.
                    layer_outputs = layer(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=causal_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        position_ids=position_ids,
                        past_key_value=layer_cache,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
                else:
                    # Standard decoder layers use only the self-attention cache.
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=layer_cache,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class Qwen2FidDecoderForCausalLM(Qwen2FidDecoderPretrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2FidDecoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Input token IDs.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the decoder.
            position_ids (`torch.LongTensor`, *optional*):
                Positions of each input token in the sequence.
            past_key_values (`Cache` or `List[torch.FloatTensor]`, *optional*):
                Past key values for caching.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Embedded representations of input tokens.
            labels (`torch.LongTensor`, *optional*):
                Labels for computing the loss.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Hidden states from the encoder.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the encoder hidden states.
            use_cache (`bool`, *optional*):
                Whether or not to use past key values.
            output_attentions (`bool`, *optional*):
                Whether or not to return attentions.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return hidden states.
            return_dict (`bool`, *optional*):
                Whether or not to return a dict.
            cache_position (`torch.LongTensor`, *optional*):
                Positions to use for caching.
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, only compute logits for the last `logits_to_keep` tokens.
                If 0, compute logits for all tokens.
            **kwargs:
                Additional keyword arguments.
                
        Returns:
            A tuple or `CausalLMOutputWithPast` containing the logits, past key values, and optionally hidden states and attentions.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logger.debug("====== Qwen2FidDecoderForCausalLM forward ======")
        logger.debug(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        logger.debug(f"use_cache: {use_cache}")

        # Forward pass through the FID decoder model (which expects encoder inputs).
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Compute logits for the necessary tokens.
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Override the method in GenerationMixin, because we use a custom cache
    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        *args,
        **kwargs
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """

        cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        # Quick escape route 1: if the user specifies a cache, we only need to:
        # a) check for conflicting `generate` arguments
        # b) convert to the new cache format (if the user passes a legacy cache and model supports it)
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                    "Cache object) is unsupported. Please use only one of the two."
                )
            if isinstance(user_defined_cache, tuple) and self._supports_default_dynamic_cache():
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(user_defined_cache)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(user_defined_cache)
                )
            return

        # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
        # `generation_config.validate()`)
        if generation_config.use_cache is False:
            return

        # Quick escape route 3: model that only supports legacy caches = nothing to prepare
        if not self._supports_default_dynamic_cache():
            if generation_config.cache_implementation is not None:
                warnings.warn(
                    "This model does not support `Cache` instances, it only supports the legacy cache format (tuple "
                    f"of tuples). `cache_implementation` (set to {generation_config.cache_implementation}) will be "
                    "ignored.",
                    UserWarning,
                )
            return

        # Otherwise we NEED to prepare a cache, based on `generation_config.cache_implementation`

        return (DynamicCache(), DynamicCache())