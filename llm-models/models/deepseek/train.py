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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import Transformer, ModelArgs, Linear
from .kernel import act_quant, weight_dequant
from .data import TinyStoriesTokenizedDataset

from torch.profiler import profile, ProfilerActivity, record_function
import deepspeed

from enum import Enum

import logging
import socket
from datetime import datetime, timedelta

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
    
    # Distributed
    ddp: bool = True
    
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

class Trainer:
    """Main training class for DeepSeek model."""
    
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.setup_distributed()
        self.setup_model()
        self.setup_optimizer()
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
        """Initialize distributed training."""
        deepspeed.init_distributed()

        if self.args.ddp:
            assert torch.cuda.is_available(), "CUDA required for distributed training"

            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_main_process = self.rank == 0
    
    def setup_model(self):
        """Initialize model and move to device."""
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(42 + self.rank)
        
        self.model = Transformer(self.args.model_args)
        self.model.to(self.device)
        
        if self.args.ddp and self.world_size > 1:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.is_main_process:
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for biases and layer norms
            if 'bias' in name or 'norm' in name or param.dim() == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps
        )
        
        # GradScaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp 
                and torch.get_default_dtype() != torch.bfloat16 )
    
    def setup_data(self):
        """Initialize data loaders."""
        train_dataset = TinyStoriesTokenizedDataset('train')
        val_dataset = TinyStoriesTokenizedDataset('validation')
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.args.ddp else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
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
        
        self.train_sampler = train_sampler
    
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

        moe_aux_loss = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, MOELayer) and hasattr(module, '_moe_aux_loss'):
                moe_aux_loss += module._moe_aux_loss

        return moe_aux_loss

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute a single training step."""
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda',enabled=self.args.use_amp, dtype=torch.bfloat16):
            # For training, we need to modify the model to return loss
            with record_function("## forward ##"):
                logits = self.model(x, start_pos=0)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1
                )
                aux_loss = self.collect_moe_aux_loss()
                loss = (loss + aux_loss) / self.args.gradient_accumulation_steps

        # Backward pass
        with record_function("## backward ##"):
            self.scaler.scale(loss).backward()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda',enabled=self.args.use_amp, dtype=torch.bfloat16):
                logits = self.model(x, start_pos=0)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 100:  # Limit eval batches
                break
        
        avg_loss = total_loss / num_batches
        
        # Aggregate across processes
        if self.args.ddp:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return avg_loss
    
    def train(self, backend='inductor', run_profile=False):
        """Main training loop."""

        running_loss = 0.0
        self.trainer.model.compile(backend=backend)
        start_time = time.time()
        
        # Initialize optimizer state
        self.optimizer.zero_grad(set_to_none=True)

        if run_profile:
            self.prof.start()

        for epoch in range(1000):  # Large number, will stop at max_steps
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            for step, batch in enumerate(self.train_loader):
                # Update learning rate
                lr = self.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                
                # Update weights after accumulation steps
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip
                    )
                    
                    # Optimizer step
                    with record_function("## optimizer ##"):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                    self.tokens_seen += (
                        self.args.batch_size * 
                        self.args.gradient_accumulation_steps * 
                        self.args.model_args.max_seq_len * 
                        self.world_size
                    )
                    
                    # Logging
                    if self.global_step % self.args.log_interval == 0 and self.is_main_process:
                        avg_loss = running_loss / self.args.log_interval
                        elapsed = time.time() - start_time
                        tokens_per_sec = self.tokens_seen / elapsed
                        
                        print(f"Step {self.global_step} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Tokens/sec: {tokens_per_sec:.0f}")
                        
                        running_loss = 0.0

                    if self.global_step +1 == 80:
                        return

                    # Evaluation
                    if self.global_step % self.args.eval_interval == 0:
                        val_loss = self.evaluate()
                        if self.is_main_process:
                            print(f"Validation loss: {val_loss:.4f}")
                            
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_checkpoint('best')
                    
                    # Checkpointing
                    if self.global_step % self.args.save_interval == 0 and self.is_main_process:
                        self.save_checkpoint(f'step_{self.global_step}')
                    
                    # Check if training is complete
                    if self.global_step >= self.args.max_steps:
                        if self.is_main_process:
                            print("Training complete!")
                            self.save_checkpoint('final')
                        return

                self.prof.step()

            self.prof.stop()

    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'{name}.pt')
        
        model_state = self.model.module.state_dict() if self.args.ddp else self.model.state_dict()
        
        checkpoint = {
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
            'tokens_seen': self.tokens_seen,
            'best_val_loss': self.best_val_loss,
            'args': enum_to_value(asdict(self.args))
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.args.ddp:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.global_step = checkpoint['global_step']
        self.tokens_seen = checkpoint['tokens_seen']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")
    
    def save_config(self):
        """Save training configuration."""
        config_path = os.path.join(self.args.checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(enum_to_value(asdict(self.args)), f, indent=2)


def main():
    """Main entry point for training."""
    # Initialize training arguments
    model_args = ModelArgs(
        max_batch_size=8,
        max_seq_len=2048,
        dtype="bf16",
        vocab_size=102400,
        dim=2048,
        n_layers=27,
    )
    
    training_args = TrainingArgs(
        model_args=model_args,
        batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=100000,
        learning_rate=3e-4,
        checkpoint_dir="./checkpoints",
        data_dir="./data"
    )
    
    # Initialize trainer
    trainer = Trainer(training_args)
    
    # Start training
    trainer.train()
    
    # Cleanup
    if training_args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()