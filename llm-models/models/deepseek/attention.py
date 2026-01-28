import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, Union, Literal
import dataclasses
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from dataclasses import dataclass, field

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)

from megatron.core.transformer.utils import (
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core import tensor_parallel

from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params, is_te_min_version
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


@dataclass
class BidirMLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention_backward: Union[ModuleSpec, type] = None
    core_attention_forward: Union[ModuleSpec, type] = None
    core_attention_forward_to_backward: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None

@dataclass
class AttentionKey:
    forward: Tensor
    backward: Tensor

@dataclass
class AttentionQuery:
    forward: Tensor
    backward: Tensor

@dataclass
class AttentionValue:
    forward: Tensor
    backward: Tensor


class BidirMLASelfAttention(MLASelfAttention):

    def __init__(self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
        ):
            super().__init__(config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            )
    

    def _checkpointed_attention_forward(
        self,
        query: AttentionQuery,
        key: AttentionKey,
        value: AttentionValue,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            assert isinstance(query, AttentionQuery)
            assert isinstance(key, AttentionKey)
            assert isinstance(value, AttentionValue)

            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_backward_to_backward_ = self.core_attention_backward(
                query.backward,
                key.backward,
                value.backward,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            output_forward_to_backward_ = self.core_attention_forward_to_backward(
                query.backward,
                (key.forward+key.backward)/2,
                (value.forward+value.backward)/2,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            output_backward_ = (output_backward_to_backward_+output_forward_to_backward_)/2

            output_forward_ = self.core_attention_forward(
                (query.forward+query.backward)/2,
                (key.forward+key.backward)/2,
                (value.forward+value.backward)/2,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

            return output_backward_, output_forward_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states_backward, hidden_states_forward = tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask, rotary_pos_emb, attn_mask_type
        )



        return hidden_states_backward, hidden_states_forward
    
    def forward(
        self,
        hidden_states_backward,
        hidden_states_forward,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for bidirectional multi-latent attention.
        In order to avoid token self attention in bidirectional model, we have to keep backward attention hidden states and
        forward attention hidden staets. Otherwise, information for any token will flow to previous tokens and then back to current token 
        and this causes contamination.
        """
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query_backward, key_backward, value_backward = self.get_query_key_value_tensors(
            hidden_states_forward,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )
        query_forward, key_forward, value_forward = self.get_query_key_value_tensors(
            hidden_states_backward,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        inference_context, query_backward, key_backward, _, attn_mask_type_backward, block_table_backward = self._adjust_key_value_for_inference(
            inference_context, query_backward, key_backward, value_backward, rotary_pos_emb=None
        )
        inference_context, query_forward, key_forward, _, attn_mask_type_backward, block_table_backward = self._adjust_key_value_for_inference(
            inference_context, query_forward, key_forward, value_forward, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query_backward, query_forward = query_backward.contiguous(), query_forward.contiguous()
        key_backward, key_forward = key_backward.contiguous(), key_forward.contiguous()

        # Value is none during decode for absorption
        if value_backward is not None:
            value_backward, value_forward = value_backward.contiguous(), value_forward.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out_backward, core_attn_out_ = self._checkpointed_attention_forward(
                AttentionQuery(forward=query_forward, backward=query_backward),
                AttentionKey(forward=key_forward, backward=key_backward),
                AttentionValue(forward=value_forward, backward=value_backward),
                attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=attn_mask_type,
                )
            elif self.cache_mla_latents:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

AttentionMode = Literal['backward', 'forward']

class FlexDotProductAttention(torch.nn.Module):
    """Wrapper that uses PyTorch's flex_attention as a drop-in replacement for 
    Transformer-Engine's DotProductAttention.
    
    Note: This implementation focuses on core functionality and may need additional
    features depending on your specific use case.
    """
    
    cp_stream: Optional[torch.Stream] = None
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        look_forward_num_tokens: int = 10,
        mode: AttentionMode = 'backward',
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__()
        
        self.config = config
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.qkv_format: str = "sbhd"
        
        # Set attention parameters
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = getattr(config, 'num_query_groups', config.num_attention_heads)
        self.kv_channels = k_channels if k_channels is not None else config.kv_channels
        self.v_channels = v_channels if v_channels is not None else config.kv_channels
        
        # set how many tokens to look forward in forward attention
        self.look_forward_num_tokens = look_forward_num_tokens

        # forward mode: current token attends to all preceding tokens and following tokens of self.layer_number*self.look_forward_num_tokens tokens downstream
        # backward mode: current token attends to preceding tokens that are at least self.layer_number*self.look_forward_num_tokens tokens upstream 
        # to avoid contamination
        self.mode = mode

        # Dropout
        self.attention_dropout = (
            config.attention_dropout if attention_dropout is None else attention_dropout
        )
        self.dropout = torch.nn.Dropout(self.attention_dropout) if self.attention_dropout > 0 else None
        
        # Softmax scale
        if softmax_scale is not None:
            self.softmax_scale = softmax_scale
        else:
            self.softmax_scale = 1.0 / (self.kv_channels ** 0.5)
        
        # Window attention
        self.window_size = None
        if is_layer_window_attention(
            config.window_size, 
            getattr(config, 'window_attn_skip_freq', None), 
            layer_number
        ):
            self.window_size = config.window_size
        
        # Context parallelism setup
        self.use_cp = config.context_parallel_size > 1
        if self.use_cp:
            if FlexDotProductAttention.cp_stream is None:
                FlexDotProductAttention.cp_stream = torch.cuda.Stream()
            self.cp_stream = FlexDotProductAttention.cp_stream
        
        # Process group collection
        if pg_collection is None:
            self.tp_group = get_tensor_model_parallel_group(check_initialized=False)
            self.cp_group = get_context_parallel_group(check_initialized=False)
        else:
            self.tp_group = pg_collection.tp
            self.cp_group = getattr(pg_collection, 'cp', None)
        
        # Packed sequence parameters tracking
        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )
    
    def _create_attention_mask(
        self,
        query: Tensor,
        key: Tensor,
        attn_mask_type: AttnMaskType,
        attention_mask: Optional[Tensor] = None,
        qkv_format: str = "sbhd",
    ):
        """Create mask function for flex_attention."""
        
        def no_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            return Tensor(True)
        
        def causal_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor)-> Tensor:
            return q_idx >= kv_idx
        
        def causal_bottom_right_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor)-> Tensor:
            # For inference with single token, allow attending to all past tokens
            return q_idx >= kv_idx
        
        def sliding_window_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor):
            # Sliding window attention
            causal_condition = q_idx >= kv_idx
            window_condition = (q_idx - kv_idx) <= Tensor(self.window_size)
            return causal_condition & window_condition
        
        # Select mask function based on type
        if attn_mask_type == AttnMaskType.no_mask:
            if self.window_size is not None:
                mask_fn = sliding_window_mask
            else:
                mask_fn = no_mask
        elif attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal):
            if self.window_size is not None:
                mask_fn = sliding_window_mask
            else:
                mask_fn = causal_mask
        elif attn_mask_type == AttnMaskType.causal_bottom_right:
            mask_fn = causal_bottom_right_mask
        else:
            mask_fn = no_mask
        
        return mask_fn
    
    def _handle_gqa(self, query: Tensor, key: Tensor, value: Tensor, qkv_format: str):
        """Handle Grouped Query Attention by repeating K/V heads if needed."""
        if self.num_query_groups == self.num_attention_heads:
            return query, key, value
        
        # Determine dimensions based on format
        if qkv_format == "sbhd":
            head_dim = 2
        elif qkv_format == "bshd":
            head_dim = 2
        elif qkv_format == "thd":
            head_dim = 1
        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")
        
        # Repeat K and V to match number of query heads
        num_repeats = self.num_attention_heads // self.num_query_groups
        key = key.repeat_interleave(num_repeats, dim=head_dim)
        value = value.repeat_interleave(num_repeats, dim=head_dim)
        
        return query, key, value
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward pass using flex_attention."""
        
        # Extract packed sequence parameters
        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)
        
        # Handle window attention mask type adjustment for inference
        if attn_mask_type == AttnMaskType.no_mask and self.window_size is not None:
            if (qkv_format == "bshd" and query.size(1) == 1) or (
                qkv_format == "sbhd" and query.size(0) == 1
            ):
                attn_mask_type = AttnMaskType.causal_bottom_right
        
        # Convert to BSHD format if needed (flex_attention expects batch-first)
        original_format = qkv_format
        if qkv_format == "sbhd":
            # [S, B, H, D] -> [B, H, S, D]
            query = query.permute(1, 2, 0, 3)
            key = key.permute(1, 2, 0, 3)
            value = value.permute(1, 2, 0, 3)
        elif qkv_format == "thd":
            raise NotImplementedError("THD format requires additional handling for flex_attention")
        elif qkv_format != "bshd":
            # [B, S, H, D] -> [B, H, S, D]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        
        # Handle GQA
        query, key, value = self._handle_gqa(query, key, value, "bshd")
        
        # Create mask function
        mask_fn = self._create_attention_mask(query, key, attn_mask_type, attention_mask, "bshd")
        
        # Create block mask for efficient computation
        B, H, S_q, D = query.shape
        S_kv = key.size(2)
        block_mask = create_block_mask(mask_fn, B, H, S_q, S_kv)
        
        # Apply flex_attention
        flex_attention_compiled = torch.compile(flex_attention)
        output, _ = flex_attention_compiled(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=(self.num_query_groups != self.num_attention_heads)
        )
        
        # Apply attention bias if provided
        if attention_bias is not None:
            # Attention bias would need to be integrated into the mask_fn or applied post-softmax
            # For simplicity, we'll add it as a score modifier
            raise NotImplementedError("attention_bias support requires custom score_mod function")
        
        # Apply dropout
        if self.dropout is not None and self.training:
            output = self.dropout(output)
        
        # Convert back to original format
        if original_format == "sbhd":
            # [B, H, S, D] -> [S, B, H, D]
            output = output.permute(2, 0, 1, 3)
        elif original_format != "bshd":
            # [B, H, S, D] -> [B, S, H, D]
            output = output.transpose(1, 2)
        
        return output