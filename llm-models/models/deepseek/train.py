import os
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from .model import Transformer, ModelArgs, Linear
from .kernel import act_quant, weight_dequant
from .data import TinyStoriesTokenizedDataset

from torch.profiler import profile, ProfilerActivity, record_function
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from models.deepseek.model import MoE
from models.deepseek import data
from models.deepseek import model

from flash_attn.losses.cross_entropy import CrossEntropyLoss

from enum import Enum

import logging
import socket
from datetime import datetime, timedelta

train_dtype = torch.bfloat16

@dataclass
class TrainingArgs:
    """Training configuration parameters."""
    # Model
    model_args: ModelArgs = None # type: ignore
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 100000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Optimization
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    checkpoint_dir: str = "./checkpoints"
    
    # Data
    data_dir: str = "./data"
    num_workers: int = 4
    
    # Mixed precision
    use_amp: bool = True
    
    # DeepSpeed
    deepspeed_config: Optional[str] = None
    local_rank: int = -1
    
    def __post_init__(self):
        if self.model_args is None:
            self.model_args = ModelArgs()


def enum_to_value(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: enum_to_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [enum_to_value(v) for v in obj]
    return obj

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.

   TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


def get_deepspeed_config(args: TrainingArgs) -> dict:
    """Generate DeepSpeed configuration."""
    config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.grad_clip,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [args.beta1, args.beta2],
                "eps": args.eps,
                "weight_decay": args.weight_decay
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.max_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
            }
        },
        
        "fp16": {
            "enabled": args.use_amp and train_dtype == torch.float16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "bf16": {
            "enabled": args.use_amp and train_dtype == torch.bfloat16
        },
        
        "zero_optimization": {
            "stage": 2,  # Stage 2 partitions optimizer states + gradients
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        
        "wall_clock_breakdown": False,
        "steps_per_print": args.log_interval,
    }
    
    return config


class Trainer:
    """Main training class for DeepSeek model with DeepSpeed."""
    
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        self.setup_profiler()

        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float('inf')
        
        if self.is_main_process:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            self.save_config()
    
    def setup_profiler(self):
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            # with_stack=True,
            # on_trace_ready=trace_handler
            )

    def setup_distributed(self):
        """Initialize distributed training with DeepSpeed."""
        # DeepSpeed handles distributed initialization
        deepspeed.init_distributed()
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        self.is_main_process = self.rank == 0
    
    def setup_model(self):
        """Initialize model with DeepSpeed."""
        torch.set_default_dtype(train_dtype)
        torch.manual_seed(42 + self.rank)
        
        # Create model
        model = Transformer(self.args.model_args)
        
        # Load or create DeepSpeed config
        if self.args.deepspeed_config and os.path.exists(self.args.deepspeed_config):
            with open(self.args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            ds_config = get_deepspeed_config(self.args)
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
        )
        
        self.model = self.model_engine.module
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.is_main_process:
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"DeepSpeed ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 0)}")
    
    def setup_data(self):
        """Initialize data loaders."""
        train_dataset = TinyStoriesTokenizedDataset('train')
        val_dataset = TinyStoriesTokenizedDataset('validation')
        
        # DeepSpeed doesn't require manual DistributedSampler
        # It handles data distribution internally
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def get_lr(self, step: int) -> float:
        """Calculate learning rate with warmup and cosine decay."""
        if step < self.args.warmup_steps:
            # Linear warmup
            return self.args.learning_rate * step / self.args.warmup_steps
        elif step < self.args.max_steps:
            # Cosine decay
            decay_ratio = (step - self.args.warmup_steps) / (self.args.max_steps - self.args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.args.min_learning_rate + coeff * (self.args.learning_rate - self.args.min_learning_rate)
        else:
            return self.args.min_learning_rate
    
    def collect_moe_aux_loss(self) -> float:
        """Collect auxiliary loss from MoE layers."""
        moe_aux_loss = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, MoE) and hasattr(module, '_moe_aux_loss') \
                    and module._moe_aux_loss is not None:
                moe_aux_loss += module._moe_aux_loss

        return moe_aux_loss

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute a single training step with DeepSpeed."""
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        
        aux_loss_w = 0.01
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda',enabled=self.args.use_amp, dtype=train_dtype):
            # For training, we need to modify the model to return loss
            with record_function("## forward ##"):
                logits = self.model(x, start_pos=0)
                criterion = CrossEntropyLoss(inplace_backward=True)
                loss = criterion(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), 
                        y[:, 1:].contiguous().view(-1))

            aux_loss = self.collect_moe_aux_loss()
            if self.is_main_process and self.global_step % self.args.log_interval == 0:
                print(f'aux_loss: {aux_loss}')
            
            loss = loss + aux_loss * aux_loss_w

        # Backward pass - DeepSpeed handles gradient accumulation
        with record_function("## backward ##"):
            self.model_engine.backward(loss)
        
        # Optimizer step - DeepSpeed handles gradient clipping and accumulation
        with record_function("## optimizer ##"):
            self.model_engine.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model on validation set."""
        self.model_engine.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            logits = self.model_engine(x, start_pos=0)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                y[:, 1:].contiguous().view(-1),
                ignore_index=-1
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 100:  # Limit eval batches
                break
        
        avg_loss = total_loss / num_batches
        
        # Aggregate across processes
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        self.model_engine.train()
        
        return avg_loss
    
    def train(self, run_profile=False):
        """Main training loop with DeepSpeed."""
        running_loss = 0.0
        start_time = time.time()

        if run_profile:
            self.prof.start()

        epoch = 0
        while self.global_step < self.args.max_steps:
            if self.is_main_process:
                print(f"\n=== Epoch {epoch} ===")
            
            for step, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                
                self.global_step += 1
                self.tokens_seen += (
                    self.args.batch_size * 
                    self.args.gradient_accumulation_steps * 
                    self.args.model_args.max_seq_len * 
                    self.world_size
                )
                
                # Get current learning rate from scheduler
                current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.get_lr(self.global_step)
                
                # Logging
                if self.global_step % self.args.log_interval == 0 and self.is_main_process:
                    avg_loss = running_loss / self.args.log_interval
                    elapsed = time.time() - start_time
                    tokens_per_sec = self.tokens_seen / elapsed
                    
                    print(f"Step {self.global_step} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Tokens/sec: {tokens_per_sec:.0f}")
                    
                    running_loss = 0.0

                # Evaluation
                if self.global_step % self.args.eval_interval == 0:
                    val_loss = self.evaluate()
                    if self.is_main_process:
                        print(f"Validation loss: {val_loss:.4f}")
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint('best')
                
                # Checkpointing
                if self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
                
                # Check if training is complete
                if self.global_step >= self.args.max_steps:
                    if self.is_main_process:
                        print("Training complete!")
                    self.save_checkpoint('final')
                    if run_profile:
                        self.prof.stop()
                    return

                if run_profile:
                    self.prof.step()
            
            epoch += 1
        
        if run_profile:
            self.prof.stop()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint using DeepSpeed."""
        if self.is_main_process:
            checkpoint_dir = os.path.join(self.args.checkpoint_dir, name)
            
            # DeepSpeed saves distributed checkpoints
            self.model_engine.save_checkpoint(checkpoint_dir)
            
            # Save additional training state
            extra_state = {
                'global_step': self.global_step,
                'tokens_seen': self.tokens_seen,
                'best_val_loss': self.best_val_loss,
                'args': enum_to_value(asdict(self.args))
            }
            
            extra_state_path = os.path.join(checkpoint_dir, 'trainer_state.pt')
            torch.save(extra_state, extra_state_path)
            
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint using DeepSpeed."""
        # Load DeepSpeed checkpoint
        _, client_state = self.model_engine.load_checkpoint(checkpoint_dir)
        
        # Load additional training state
        extra_state_path = os.path.join(checkpoint_dir, 'trainer_state.pt')
        if os.path.exists(extra_state_path):
            extra_state = torch.load(extra_state_path, map_location=self.device)
            self.global_step = extra_state['global_step']
            self.tokens_seen = extra_state['tokens_seen']
            self.best_val_loss = extra_state['best_val_loss']
        
        if self.is_main_process:
            print(f"Loaded checkpoint from {checkpoint_dir}")
            print(f"Resuming from step {self.global_step}")
    
    def save_config(self):
        """Save training configuration."""
        config_path = os.path.join(self.args.checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(enum_to_value(asdict(self.args)), f, indent=2)


def main():
    """Main entry point for training."""
    # Parse DeepSpeed arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='Path to DeepSpeed config file')
    args = parser.parse_args()
    
    # Initialize training arguments
    scale = 2
    model_args = ModelArgs(
        expert_type=model.ExpertType(1),
        dim=128*scale, 
        inter_dim=341*scale, 
        moe_inter_dim=64*scale,
        kv_lora_rank=32*scale, 
        qk_nope_head_dim=8*scale, 
        qk_rope_head_dim=4*scale, 
        v_head_dim=8*scale, 
        n_routed_experts=32, 
        n_layers=4*scale,
        original_seq_len=data.CONTEXT_LENGTH
    )
    
    training_args = TrainingArgs(
        model_args=model_args,
        batch_size=8,
        gradient_accumulation_steps=16,
        max_steps=100000,
        learning_rate=3e-4,
        checkpoint_dir="./checkpoints",
        data_dir="./data",
        deepspeed_config=args.deepspeed_config,
        local_rank=args.local_rank
    )
    
    # Initialize trainer
    trainer = Trainer(training_args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
