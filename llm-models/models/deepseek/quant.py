"""
Post-training quantization utilities for DeepSeek models.
Converts trained BF16 models to FP8 for efficient inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm
import os

from model import Transformer, ModelArgs, Linear
from kernel import act_quant


class Quantizer:
    """
    Handles model quantization from BF16 to FP8.
    """
    def __init__(self, block_size: int = 128, scale_fmt: Optional[str] = None):
        self.block_size = block_size
        self.scale_fmt = scale_fmt
    
    def quantize_weight(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a weight tensor to FP8 with per-block scaling.
        
        Args:
            weight: BF16 weight tensor of shape (out_features, in_features)
        
        Returns:
            Tuple of (quantized_weight, scale_tensor)
        """
        assert weight.dim() == 2, "Weight must be 2D"
        assert weight.is_contiguous(), "Weight must be contiguous"
        
        out_features, in_features = weight.shape
        
        # Pad to block size
        scale_out = (out_features + self.block_size - 1) // self.block_size
        scale_in = (in_features + self.block_size - 1) // self.block_size
        
        # Initialize quantized weight and scale
        q_weight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        scale = torch.empty(scale_out, scale_in, dtype=torch.float32, device=weight.device)
        
        # Quantize block by block
        for i in range(scale_out):
            for j in range(scale_in):
                out_start = i * self.block_size
                out_end = min(out_start + self.block_size, out_features)
                in_start = j * self.block_size
                in_end = min(in_start + self.block_size, in_features)
                
                block = weight[out_start:out_end, in_start:in_end].float()
                
                # Compute scale
                amax = block.abs().max()
                amax = max(amax.item(), 1e-4)
                s = amax / 448.0  # FP8 E4M3 max value
                
                if self.scale_fmt == "ue8m0":
                    # Use power-of-2 scaling
                    import math
                    exp = math.ceil(math.log2(s))
                    s = 2 ** exp
                
                # Quantize block
                q_block = (block / s).to(torch.float8_e4m3fn)
                q_weight[out_start:out_end, in_start:in_end] = q_block
                scale[i, j] = s
        
        return q_weight, scale
    
    def quantize_linear_layer(self, layer: Linear) -> None:
        """
        Quantize a Linear layer in-place.
        
        Args:
            layer: Linear layer to quantize
        """
        if layer.weight.element_size() == 1:
            # Already quantized
            return
        
        with torch.no_grad():
            q_weight, scale = self.quantize_weight(layer.weight.data)
            
            # Replace weight with quantized version
            layer.weight.data = q_weight
            
            # Add scale parameter
            if not hasattr(layer, 'scale') or layer.scale is None:
                layer.scale = nn.Parameter(scale, requires_grad=False)
            else:
                layer.scale.data = scale
    
    def quantize_model(self, model: Transformer, verbose: bool = True) -> Transformer:
        """
        Quantize all linear layers in the model.
        
        Args:
            model: Transformer model to quantize
            verbose: Whether to print progress
        
        Returns:
            Quantized model
        """
        linear_layers = []
        
        # Collect all Linear layers
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                linear_layers.append((name, module))
        
        if verbose:
            print(f"Found {len(linear_layers)} linear layers to quantize")
        
        # Quantize each layer
        for name, layer in tqdm(linear_layers, desc="Quantizing layers", disable=not verbose):
            self.quantize_linear_layer(layer)
        
        if verbose:
            print("Quantization complete!")
        
        return model


def analyze_quantization_error(
    original_model: Transformer,
    quantized_model: Transformer,
    test_input: torch.Tensor,
    num_samples: int = 100
) -> dict:
    """
    Analyze the error introduced by quantization.
    
    Args:
        original_model: Original BF16 model
        quantized_model: Quantized FP8 model
        test_input: Sample input tensor
        num_samples: Number of forward passes to analyze
    
    Returns:
        Dictionary with error statistics
    """
    original_model.eval()
    quantized_model.eval()
    
    errors = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Analyzing error"):
            # Forward pass through both models
            original_output = original_model(test_input)
            quantized_output = quantized_model(test_input)
            
            # Compute error metrics
            mse = ((original_output - quantized_output) ** 2).mean().item()
            mae = (original_output - quantized_output).abs().mean().item()
            max_error = (original_output - quantized_output).abs().max().item()
            
            # Relative error
            relative_error = (
                (original_output - quantized_output).abs() / 
                (original_output.abs() + 1e-8)
            ).mean().item()
            
            errors.append({
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'relative_error': relative_error
            })
    
    # Aggregate statistics
    stats = {
        'mse': sum(e['mse'] for e in errors) / len(errors),
        'mae': sum(e['mae'] for e in errors) / len(errors),
        'max_error': max(e['max_error'] for e in errors),
        'relative_error': sum(e['relative_error'] for e in errors) / len(errors)
    }
    
    return stats


def calibrate_quantization(
    model: Transformer,
    calibration_data: torch.utils.data.DataLoader,
    num_batches: int = 100
) -> dict:
    """
    Calibrate quantization by collecting activation statistics.
    This can be used for more sophisticated quantization schemes.
    
    Args:
        model: Model to calibrate
        calibration_data: DataLoader with calibration data
        num_batches: Number of batches to use
    
    Returns:
        Dictionary with calibration statistics
    """
    model.eval()
    
    activation_stats = {}
    
    def collect_stats(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            
            # Collect min/max statistics
            activation_stats[name].append({
                'min': output.min().item(),
                'max': output.max().item(),
                'mean': output.mean().item(),
                'std': output.std().item()
            })
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            hook = module.register_forward_hook(collect_stats(name))
            hooks.append(hook)
    
    # Run calibration
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_data, desc="Calibrating", total=num_batches)):
            if i >= num_batches:
                break
            
            x, _ = batch
            x = x.to(next(model.parameters()).device)
            model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Aggregate statistics
    aggregated_stats = {}
    for name, stats_list in activation_stats.items():
        aggregated_stats[name] = {
            'min': min(s['min'] for s in stats_list),
            'max': max(s['max'] for s in stats_list),
            'mean': sum(s['mean'] for s in stats_list) / len(stats_list),
            'std': sum(s['std'] for s in stats_list) / len(stats_list)
        }
    
    return aggregated_stats


def save_quantized_model(
    model: Transformer,
    output_path: str,
    model_args: ModelArgs
):
    """
    Save quantized model to disk.
    
    Args:
        model: Quantized model
        output_path: Path to save model
        model_args: Model configuration
    """
    checkpoint = {
        'model': model.state_dict(),
        'model_args': vars(model_args),
        'quantized': True,
        'block_size': 128
    }
    
    torch.save(checkpoint, output_path)
    print(f"Saved quantized model to {output_path}")
    
    # Print model size
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")


def load_quantized_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[Transformer, ModelArgs]:
    """
    Load quantized model from disk.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (model, model_args)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model args
    model_args = ModelArgs(**checkpoint['model_args'])
    model_args.dtype = "fp8"
    
    # Create model
    model = Transformer(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, model_args


def main():
    """Example usage of quantization utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantize DeepSeek model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input BF16 model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save quantized model')
    parser.add_argument('--block-size', type=int, default=128,
                       help='Block size for quantization')
    parser.add_argument('--scale-fmt', type=str, default=None,
                       choices=['ue8m0', None],
                       help='Scale format for quantization')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze quantization error')
    
    args = parser.parse_args()
    
    # Load original model
    print(f"Loading model from {args.input}...")
    checkpoint = torch.load(args.input, map_location='cuda')
    model_args = ModelArgs(**checkpoint['model_args'])
    
    original_model = Transformer(model_args)
    original_model.load_state_dict(checkpoint['model'])
    original_model.cuda()
    original_model.eval()
    
    # Create quantizer
    quantizer = Quantizer(block_size=args.block_size, scale_fmt=args.scale_fmt)
    
    # Quantize model
    print("Quantizing model...")
    quantized_model = quantizer.quantize_model(original_model)
    
    # Analyze error if requested
    if args.analyze:
        print("\nAnalyzing quantization error...")
        test_input = torch.randint(0, model_args.vocab_size, (4, 128), device='cuda')
        error_stats = analyze_quantization_error(original_model, quantized_model, test_input)
        
        print("\nQuantization Error Statistics:")
        print(f"  MSE: {error_stats['mse']:.6f}")
        print(f"  MAE: {error_stats['mae']:.6f}")
        print(f"  Max Error: {error_stats['max_error']:.6f}")
        print(f"  Relative Error: {error_stats['relative_error']:.4%}")
    
    # Save quantized model
    print(f"\nSaving quantized model to {args.output}...")
    save_quantized_model(quantized_model, args.output, model_args)
    
    print("Done!")


if __name__ == '__main__':
    main()