# pyright: reportArgumentType=false
# pyright: reportOptionalOperand=false
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, Union, Literal
import dataclasses
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from dataclasses import dataclass, field
import enum

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
class BidirMLASelfAttentionSubmodules: # type: ignore
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None  # type: ignore
    linear_q_down_proj: Union[ModuleSpec, type] = None # type: ignore
    linear_q_up_proj: Union[ModuleSpec, type] = None # type: ignore
    linear_kv_down_proj: Union[ModuleSpec, type] = None # type: ignore
    linear_kv_up_proj: Union[ModuleSpec, type] = None # type: ignore
    core_attention_backward: Union[ModuleSpec, type] = None # type: ignore
    bidir_attention_forward: Union[ModuleSpec, type] = None # type: ignore
    bidir_attention_backward: Union[ModuleSpec, type] = None # type: ignore
    linear_proj: Union[ModuleSpec, type] = None # type: ignore
    q_layernorm: Union[ModuleSpec, type] = None # type: ignore
    kv_layernorm: Union[ModuleSpec, type] = None # type: ignore

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

class BidirAttnMaskType(enum.Enum):
    """Attention Mask Type"""
    padding = AttnMaskType.padding.value
    causal = AttnMaskType.padding.value
    no_mask = AttnMaskType.padding.value
    padding_causal = AttnMaskType.padding.value
    arbitrary = AttnMaskType.padding.value
    causal_bottom_right = AttnMaskType.padding.value
    bidir_forward = 7
    bidir_backward = 8

class BidirMLASelfAttention(MLASelfAttention):

    config: MLATransformerConfig
    
    def __init__(self,
        config: MLATransformerConfig,
        submodules: BidirMLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        bidir_forward_attn_mask_type=BidirAttnMaskType.bidir_forward,
        bidir_backward_attn_mask_type=BidirAttnMaskType.bidir_backward,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
        ):
            # Create a compatible submodules object for parent class
            # The parent MLASelfAttention expects MLASelfAttentionSubmodules
            parent_submodules = MLASelfAttentionSubmodules(
                linear_q_proj=submodules.linear_q_proj,
                linear_q_down_proj=submodules.linear_q_down_proj,
                linear_q_up_proj=submodules.linear_q_up_proj,
                linear_kv_down_proj=submodules.linear_kv_down_proj,
                linear_kv_up_proj=submodules.linear_kv_up_proj,
                core_attention=submodules.core_attention_backward,  # Use backward as default
                linear_proj=submodules.linear_proj,
                q_layernorm=submodules.q_layernorm,
                kv_layernorm=submodules.kv_layernorm,
            )
            
            super().__init__(
                config=config,
                submodules=parent_submodules,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                cp_comm_type=cp_comm_type,
                pg_collection=pg_collection,
            )
            
            # Store bidirectional attention mask types
            self.bidir_forward_attn_mask_type = bidir_forward_attn_mask_type
            self.bidir_backward_attn_mask_type = bidir_backward_attn_mask_type
            
            # Build bidirectional-specific attention modules
            # Backward attention (for backward queries attending to backward keys/values)
            self.core_attention_backward = build_module(
                submodules.core_attention_backward,
                config=config,
                layer_number=layer_number,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )

            # Forward attention (for forward queries attending to combined keys/values)
            self.bidir_attention_forward = build_module(
                submodules.bidir_attention_forward,
                config=config,
                layer_number=layer_number,
                attn_mask_type=self.bidir_forward_attn_mask_type,
                attention_type="self",
            )
            
            # Forward-to-backward attention (for backward queries attending to combined keys/values)
            self.bidir_attention_backward = build_module(
                submodules.bidir_attention_backward,
                config=config,
                layer_number=layer_number,
                attn_mask_type=self.bidir_backward_attn_mask_type,
                attention_type="self",
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
            # this is purely backward attention. token only attends to preceding tokens.
            output_backward_to_backward_ = self.core_attention_backward(
                query.backward,
                key.backward,
                value.backward,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            # token's backward components attends to both backward and forward components. 
            # But it only attends to tokens with information older than itsef. 
            # e.g. if forward attention look aheads to 5 tokens, it only attends to tokens that
            # are positioned -5*n before itself.
            output_forward_to_backward_ = self.bidir_attention_backward(
                query.backward,
                (key.forward+key.backward)/2,
                (value.forward+value.backward)/2,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            output_backward_ = (output_backward_to_backward_+output_forward_to_backward_)/2

            # token attend to tokens generated later than itself.
            output_forward_ = self.bidir_attention_forward(
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
        hidden_states,
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

        hidden_states_backward, hidden_states_forward = torch.chunk(hidden_states, 2, dim=1)

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
        print(hidden_states_forward.shape)
        query_backward, key_backward, value_backward, _, _ = self.get_query_key_value_tensors(
            hidden_states_forward,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )
        query_forward, key_forward, value_forward, _, _ = self.get_query_key_value_tensors(
            hidden_states_backward,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )
        print(query_backward.shape)
        print(key_backward.shape)
        print(value_backward.shape)
        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query_backward, key_backward, value_backward, _, attn_mask_type_backward, block_table_backward = self._adjust_key_value_for_inference(
            inference_context, query_backward, key_backward, value_backward, rotary_pos_emb=None
        )
        query_forward, key_forward, value_forward, _, attn_mask_type_backward, block_table_backward = self._adjust_key_value_for_inference(
            inference_context, query_forward, key_forward, value_forward, rotary_pos_emb=None
        )

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query_backward, query_forward = query_backward.contiguous(), query_forward.contiguous()
        key_backward, key_forward = key_backward.contiguous(), key_forward.contiguous()

        # Value is none during decode for absorption
        if value_backward is not None and value_forward is not None:
            value_backward, value_forward = value_backward.contiguous(), value_forward.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            attn_out_backward, attn_out_forward = self._checkpointed_attention_forward(
                AttentionQuery(forward=query_forward, backward=query_backward),
                AttentionKey(forward=key_forward, backward=key_backward),
                AttentionValue(forward=value_forward, backward=value_backward),
                attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Backward attention: backward queries attend to backward keys/values
                core_attn_out_backward_to_backward = self.core_attention_backward(
                    query_backward,
                    key_backward,
                    value_backward,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=self.bidir_backward_attn_mask_type,
                )
                # Forward-to-backward attention: backward queries attend to averaged keys/values
                key_avg = (key_forward + key_backward) / 2
                value_avg = (value_forward + value_backward) / 2
                attn_out_forward_to_backward = self.bidir_attention_backward(
                    query_backward,
                    key_avg,
                    value_avg,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=self.bidir_forward_attn_mask_type,
                )
                # Combined backward output
                attn_out_backward = (core_attn_out_backward_to_backward + attn_out_forward_to_backward) / 2
                
                # Forward attention: forward queries attend to averaged keys/values
                query_avg = (query_forward + query_backward) / 2
                attn_out_forward = self.bidir_attention_forward(
                    query_avg,
                    key_avg,
                    value_avg,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=self.bidir_forward_attn_mask_type,
                )
            elif self.cache_mla_latents:
                # Dynamic batching attention kernel - not yet supported for bidirectional
                raise NotImplementedError(
                    "Dynamic batching with cache_mla_latents is not yet supported for bidirectional attention"
                )

        # Handle decode mode with absorption (not supported for bidirectional yet)
        if self.cache_mla_latents and inference_context is not None and inference_context.is_decode_only():
            raise NotImplementedError(
                "MLA latent caching with decode mode is not yet supported for bidirectional attention"
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            attn_out_backward = attn_out_backward.reshape(attn_out_backward.size(0), 1, -1)
            attn_out_forward = attn_out_forward.reshape(attn_out_forward.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            # For bidirectional, we need to handle both outputs
            self.qkv_up_checkpoint.discard_output_and_register_recompute(attn_out_forward)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        print('core_attn_out_backward')
        print(attn_out_backward.shape)
        output_backward, bias_backward = self.linear_proj(attn_out_backward)
        output_forward, bias_forward = self.linear_proj(attn_out_forward)

        output = torch.concat((output_backward, output_forward), dim=1)
        if bias_backward is not None and bias_forward is not None:
            bias = torch.concat((bias_backward, bias_forward), dim=1)
        else:
            bias = None
            
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
        attn_mask_type: Union[AttnMaskType, BidirAttnMaskType],
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
        self.layer_number = layer_number  # Store layer_number for bidirectional masks
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

        self.flex_attention = torch.compile(flex_attention)
    
    def _create_attention_mask(
        self,
        query: Tensor,
        key: Tensor,
        attn_mask_type: Union[AttnMaskType, BidirAttnMaskType],
        attention_mask: Optional[Tensor] = None,
        qkv_format: str = "sbhd",
    ):
        """Create mask function for flex_attention."""
        
        def no_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            return torch.tensor(True)
        
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
        
        def bidir_forward_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            # Forward attention: query can attend to keys up to look_forward_num_tokens ahead
            # q_idx >= kv_idx - look_forward means kv_idx <= q_idx + look_forward
            return q_idx >= kv_idx - self.look_forward_num_tokens
        
        def bidir_backward_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            # Backward attention: query can only attend to keys that are at least
            # (look_forward_num_tokens * layer_number) positions behind
            # This prevents information from flowing forward then back
            return q_idx > kv_idx + self.look_forward_num_tokens * self.layer_number

        # Select mask function based on type
        if attn_mask_type == BidirAttnMaskType.bidir_forward:
            mask_fn = bidir_forward_mask
        elif attn_mask_type == BidirAttnMaskType.bidir_backward:
            mask_fn = bidir_backward_mask
        elif attn_mask_type in (AttnMaskType.no_mask, BidirAttnMaskType.no_mask):
            if self.window_size is not None:
                mask_fn = sliding_window_mask
            else:
                mask_fn = no_mask
        elif attn_mask_type in (AttnMaskType.causal, AttnMaskType.padding_causal, 
                                 BidirAttnMaskType.causal, BidirAttnMaskType.padding_causal):
            if self.window_size is not None:
                mask_fn = sliding_window_mask
            else:
                mask_fn = causal_mask
        elif attn_mask_type in (AttnMaskType.causal_bottom_right, BidirAttnMaskType.causal_bottom_right):
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
        attn_mask_type: Union[AttnMaskType, BidirAttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward pass using flex_attention."""
        
        # Use default mask type if not provided
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        
        # Extract packed sequence parameters
        packed_seq_kwargs = (
            {key_name: getattr(packed_seq_params, key_name) for key_name in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)
        
        # Handle window attention mask type adjustment for inference
        if attn_mask_type in (AttnMaskType.no_mask, BidirAttnMaskType.no_mask) and self.window_size is not None:
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
        output = self.flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=(self.num_query_groups != self.num_attention_heads),
            kernel_options={"BLOCK_M": 16, "BLOCK_N": 16}
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
        
        new_output_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_output_shape)

        return output