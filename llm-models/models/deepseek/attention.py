import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any
import dataclasses
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

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

from megatron.core.packed_seq_params import PackedSeqParams

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