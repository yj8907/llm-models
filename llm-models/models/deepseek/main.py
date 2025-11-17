#!/usr/bin/env python3
"""
Complete training script for DeepSeek models with all features:
- Distributed training (DDP)
- Mixed precision training
- Gradient accumulation
- Checkpointing and resuming
- Weights & Biases logging (optional)
- Learning rate scheduling
- Model evaluation
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from model import ModelArgs
from train import Trainer, TrainingArgs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DeepSeek language model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--vocab-size', type=int, default=102400,
                            help='Vocabulary size')
    model_group.add_argument('--dim', type=int, default=2048,
                            help='Model dimension')
    model_group.add_argument('--n-layers', type=int, default=27,
                            help='Number of transformer layers')
    model_group.add_argument('--n-heads', type=int, default=16,
                            help='Number of attention heads')
    model_group.add_argument('--inter-dim', type=int, default=10944,
                            help='Intermediate dimension for MLP')
    model_group.add_argument('--moe-inter-dim', type=int, default=1408,
                            help='Intermediate dimension for MoE experts')
    model_group.add_argument('--n-routed-experts', type=int, default=64,
                            help='Number of routed experts')
    model_group.add_argument('--n-shared-experts', type=int, default=2,
                            help='Number of shared experts')
    model_group.add_argument('--n-activated-experts', type=int, default=6,
                            help='Number of activated experts')
    model_group.add_argument('--kv-lora-rank', type=int, default=512,
                            help='LoRA rank for key-value projections')
    model_group.add_argument('--max-seq-len', type=int, default=2048,
                            help='Maximum sequence length')
    
    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch-size', type=int, default=4,
                            help='Batch size per GPU')
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=8,
                            help='Gradient accumulation steps')
    train_group.add_argument('--max-steps', type=int, default=100000,
                            help='Maximum training steps')
    train_group.add_argument('--warmup-steps', type=int, default=2000,
                            help='Learning rate warmup steps')
    train_group.add_argument('--learning-rate', type=float, default=3e-4,
                            help='Peak learning rate')
    train_group.add_argument('--min-learning-rate', type=float, default=3e-5,
                            help='Minimum learning rate')
    train_group.add_argument('--weight-decay', type=float, default=0.1,
                            help='Weight decay coefficient')
    train_group.add_argument('--grad-clip', type=float, default=1.0,
                            help='Gradient clipping threshold')
    train_group.add_argument('--beta1', type=float, default=0.9,
                            help='Adam beta1')
    train_group.add_argument('--beta2', type=float, default=0.95,
                            help='Adam beta2')
    
    # Data
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, required=True,
                           help='Directory containing training data')
    data_group.add_argument('--num-workers', type=int, default=4,
                           help='Number of data loading workers')
    
    # Checkpointing and logging
    io_group = parser.add_argument_group('I/O Configuration')
    io_group.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                         help='Directory for saving checkpoints')
    io_group.add_argument('--resume', type=str, default=None,
                         help='Path to checkpoint to resume from')
    io_group.add_argument('--log-interval', type=int, default=10,
                         help='Logging interval (steps)')
    io_group.add_argument('--eval-interval', type=int, default=500,
                         help='Evaluation interval (steps)')
    io_group.add_argument('--save-interval', type=int, default=5000,
                         help='Checkpoint saving interval (steps)')
    
    # System
    sys_group = parser.add_argument_group('System Configuration')
    sys_group.add_argument('--no-ddp', action='store_true',
                          help='Disable distributed training')
    sys_group.add_argument('--no-amp', action='store_true',
                          help='Disable automatic mixed precision')
    sys_group.add_argument('--compile', action='store_true',
                          help='Use torch.compile for optimization')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    
    # Weights & Biases
    wandb_group = parser.add_argument_group('Weights & Biases')
    wandb_group.add_argument('--wandb', action='store_true',
                            help='Enable Weights & Biases logging')
    wandb_group.add_argument('--wandb-project', type=str, default='deepseek',
                            help='W&B project name')
    wandb_group.add_argument('--wandb-run-name', type=str, default=None,
                            help='W&B run name')
    
    return parser.parse_args()


def setup_wandb(args, trainer):
    """Initialize Weights & Biases logging."""
    try:
        import wandb
    except ImportError:
        print("Weights & Biases not installed. Install with: pip install wandb")
        return None
    
    if not trainer.is_main_process:
        return None
    
    config = {
        'model': vars(args),
        'training': {
            'batch_size': args.batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_steps': args.max_steps,
            'learning_rate': args.learning_rate,
        }
    }
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    
    return wandb


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create model arguments
    model_args = ModelArgs(
        max_batch_size=args.batch_size * (1 if args.no_ddp else torch.cuda.device_count()),
        max_seq_len=args.max_seq_len,
        dtype="bf16",
        vocab_size=args.vocab_size,
        dim=args.dim,
        inter_dim=args.inter_dim,
        moe_inter_dim=args.moe_inter_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        kv_lora_rank=args.kv_lora_rank,
    )
    
    # Create training arguments
    training_args = TrainingArgs(
        model_args=model_args,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        beta1=args.beta1,
        beta2=args.beta2,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        ddp=not args.no_ddp,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(training_args)
    
    # Setup W&B if requested
    wandb_run = None
    if args.wandb:
        wandb_run = setup_wandb(args, trainer)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            trainer.model = torch.compile(trainer.model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    
    # Print training info
    if trainer.is_main_process:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Model dimension: {model_args.dim}")
        print(f"Number of layers: {model_args.n_layers}")
        print(f"Number of heads: {model_args.n_heads}")
        print(f"Vocabulary size: {model_args.vocab_size:,}")
        print(f"Max sequence length: {model_args.max_seq_len}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * trainer.world_size}")
        print(f"Max training steps: {args.max_steps:,}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Distributed training: {not args.no_ddp}")
        print(f"Mixed precision: {not args.no_amp}")
        print(f"World size: {trainer.world_size}")
        print("="*60 + "\n")
    
    try:
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if trainer.is_main_process:
            print("Saving checkpoint before exit...")
            trainer.save_checkpoint('interrupted')
    
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        if trainer.is_main_process:
            print("Saving checkpoint before exit...")
            trainer.save_checkpoint('error')
    
    finally:
        # Cleanup
        if wandb_run is not None:
            wandb_run.finish()
        
        if training_args.ddp:
            dist.destroy_process_group()
        
        print("\nTraining complete!")


if __name__ == "__main__":
    main()