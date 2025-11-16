import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed.moe.layer import MoE
import math
from typing import Optional, Tuple

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available. Install with: pip install flash-attn --no-build-isolation")


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with Flash Attention support"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        if self.use_flash and self.training:
            # Flash Attention path - much faster and memory efficient
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Reshape to (batch, seq_len, num_heads, head_dim)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Flash attention expects (batch, seq_len, num_heads, head_dim)
            # and handles causal masking internally if specified
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                softmax_scale=1.0 / self.scale,
                causal=True  # Set to True for causal/autoregressive attention
            )
            
            # Reshape back
            attn_output = attn_output.view(batch_size, seq_len, self.d_model)
            
        else:
            # Standard attention path (fallback)
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    """Standard feed-forward network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with optional MoE and Flash Attention"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        expert_capacity: int = 32,
        top_k: int = 2,
        use_flash_attn: bool = True
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_flash=use_flash_attn)
        self.use_moe = use_moe
        
        if use_moe:
            # DeepSpeed MoE layer
            self.ffn = MoE(
                hidden_size=d_model,
                expert=FeedForward(d_model, d_ff, dropout),
                num_experts=num_experts,
                ep_size=1,  # Expert parallelism size
                k=top_k,  # Top-k routing
                capacity_factor=1.0,
                eval_capacity_factor=1.0,
                min_capacity=4,
                use_residual=False,
            )
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        if self.use_moe:
            # MoE returns (output, moe_loss, moe_logits)
            ffn_output, _, _ = self.ffn(x)
        else:
            ffn_output = self.ffn(x)
        
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers with Flash Attention"""
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        moe_frequency: int = 2,  # Apply MoE every N layers
        num_experts: int = 8,
        expert_capacity: int = 32,
        top_k: int = 2,
        use_flash_attn: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_moe=use_moe and (i % moe_frequency == 0),
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                top_k=top_k,
                use_flash_attn=use_flash_attn
            )
            for i in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerMoE(nn.Module):
    """Complete Transformer model with Mixture of Experts and Flash Attention"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_moe: bool = True,
        moe_frequency: int = 2,
        num_experts: int = 8,
        expert_capacity: int = 32,
        top_k: int = 2,
        num_classes: Optional[int] = None,
        use_flash_attn: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_moe=use_moe,
            moe_frequency=moe_frequency,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            top_k=top_k,
            use_flash_attn=use_flash_attn
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output head (for classification or language modeling)
        if num_classes is not None:
            self.output_head = nn.Linear(d_model, num_classes)
        else:
            self.output_head = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Encoder
        x = self.encoder(x, attention_mask)
        
        # Output projection
        logits = self.output_head(x)
        
        return logits


def create_deepspeed_config(
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    use_fp16: bool = True,
    use_zero: bool = True,
    zero_stage: int = 2
):
    """Create DeepSpeed configuration"""
    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000
            }
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
    
    if use_fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    
    if use_zero:
        config["zero_optimization"] = {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu" if zero_stage == 2 else "none"
            },
            "offload_param": {
                "device": "cpu" if zero_stage == 3 else "none"
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        }
    
    return config


# Example usage and training loop
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Model configuration
    model = TransformerMoE(
        vocab_size=30000,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        use_moe=True,
        moe_frequency=2,  # Every 2nd layer uses MoE
        num_experts=8,
        top_k=2,
        num_classes=None  # For language modeling
    )
    
    # DeepSpeed config
    ds_config = create_deepspeed_config(
        train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        use_fp16=True,
        use_zero=True,
        zero_stage=2
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    print(f"Model initialized with DeepSpeed")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop example
    model_engine.train()
    
    for step in range(100):
        # Dummy data
        input_ids = torch.randint(0, 30000, (8, 128)).to(model_engine.local_rank)
        labels = torch.randint(0, 30000, (8, 128)).to(model_engine.local_rank)
        
        # Forward pass
        logits = model_engine(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Save model
    model_engine.save_checkpoint("./checkpoints", tag="final")
    print("Training complete and model saved!")