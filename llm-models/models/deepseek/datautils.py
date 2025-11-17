"""
Data preparation utilities for DeepSeek training.
This module handles tokenization, data preprocessing, and dataset creation.
"""

import os
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Iterator
import numpy as np
from tqdm import tqdm


class Tokenizer:
    """
    Simple tokenizer wrapper. In practice, you would use a proper tokenizer
    like SentencePiece or the Hugging Face tokenizers library.
    """
    def __init__(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Simple whitespace tokenization (replace with proper tokenizer)
        tokens = text.split()
        return [self.token_to_id.get(token, 0) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(idx, '<unk>') for idx in ids]
        return ' '.join(tokens)


def process_text_file(args):
    """Process a single text file and return tokenized data."""
    file_path, tokenizer_path = args
    
    # Load tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    
    # Read and tokenize file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    return np.array(tokens, dtype=np.uint16)


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    split: str = 'train',
    num_workers: int = 8
):
    """
    Prepare dataset from text files.
    
    Args:
        input_dir: Directory containing text files
        output_dir: Directory to save processed data
        tokenizer_path: Path to tokenizer vocabulary
        split: Dataset split name (train/val/test)
        num_workers: Number of parallel workers
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    text_files = list(input_path.glob('*.txt'))
    print(f"Found {len(text_files)} text files")
    
    # Process files in parallel
    print("Processing files...")
    with mp.Pool(num_workers) as pool:
        args = [(str(f), tokenizer_path) for f in text_files]
        results = list(tqdm(
            pool.imap(process_text_file, args),
            total=len(text_files)
        ))
    
    # Concatenate all tokens
    print("Concatenating tokens...")
    all_tokens = np.concatenate(results)
    
    # Save to binary file
    output_file = output_path / f'{split}.bin'
    print(f"Saving to {output_file}...")
    all_tokens.tofile(str(output_file))
    
    # Save metadata
    metadata = {
        'num_tokens': len(all_tokens),
        'num_files': len(text_files),
        'dtype': str(all_tokens.dtype)
    }
    
    with open(output_path / f'{split}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset prepared: {len(all_tokens):,} tokens")


def create_vocab_from_corpus(
    corpus_files: List[str],
    output_path: str,
    vocab_size: int = 102400,
    min_frequency: int = 2
):
    """
    Create vocabulary from corpus files.
    This is a simplified version - use proper BPE/SentencePiece in production.
    
    Args:
        corpus_files: List of text file paths
        output_path: Path to save vocabulary
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency to include
    """
    from collections import Counter
    
    print("Building vocabulary...")
    token_counts = Counter()
    
    for file_path in tqdm(corpus_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            tokens = text.split()  # Simple whitespace tokenization
            token_counts.update(tokens)
    
    # Filter by frequency and take most common
    vocab_tokens = [
        token for token, count in token_counts.most_common()
        if count >= min_frequency
    ][:vocab_size]
    
    # Add special tokens
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    vocab = special_tokens + vocab_tokens
    
    # Save vocabulary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary created: {len(vocab)} tokens")
    print(f"Saved to {output_path}")


def split_dataset(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05
):
    """
    Split a dataset into train/val/test splits.
    
    Args:
        input_file: Path to input binary file
        output_dir: Directory to save splits
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
    """
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = np.fromfile(input_file, dtype=np.uint16)
    total_tokens = len(data)
    
    # Calculate split points
    train_end = int(total_tokens * train_ratio)
    val_end = train_end + int(total_tokens * val_ratio)
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save splits
    train_data.tofile(str(output_path / 'train.bin'))
    val_data.tofile(str(output_path / 'val.bin'))
    test_data.tofile(str(output_path / 'test.bin'))
    
    print(f"Train: {len(train_data):,} tokens")
    print(f"Val: {len(val_data):,} tokens")
    print(f"Test: {len(test_data):,} tokens")


class DataShuffler:
    """
    Utility to shuffle large datasets that don't fit in memory.
    Uses chunk-based shuffling for efficient processing.
    """
    def __init__(self, chunk_size: int = 1_000_000):
        self.chunk_size = chunk_size
    
    def shuffle_file(self, input_file: str, output_file: str):
        """Shuffle a binary data file."""
        data = np.fromfile(input_file, dtype=np.uint16)
        total_size = len(data)
        
        print(f"Shuffling {total_size:,} tokens...")
        
        # Shuffle in chunks
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        
        for i in tqdm(range(num_chunks)):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total_size)
            chunk = data[start:end]
            np.random.shuffle(chunk)
            data[start:end] = chunk
        
        # Final global shuffle
        np.random.shuffle(data)
        
        # Save shuffled data
        data.tofile(output_file)
        print(f"Saved shuffled data to {output_file}")


def main():
    """Example usage of data preparation utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--task', type=str, required=True,
                       choices=['vocab', 'prepare', 'split', 'shuffle'],
                       help='Task to perform')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (file or directory)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path (file or directory)')
    parser.add_argument('--tokenizer', type=str, default='vocab.json',
                       help='Path to tokenizer vocabulary')
    parser.add_argument('--vocab-size', type=int, default=102400,
                       help='Vocabulary size')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of workers for parallel processing')
    
    args = parser.parse_args()
    
    if args.task == 'vocab':
        # Create vocabulary from text files
        text_files = list(Path(args.input).glob('*.txt'))
        create_vocab_from_corpus(
            [str(f) for f in text_files],
            args.output,
            vocab_size=args.vocab_size
        )
    
    elif args.task == 'prepare':
        # Prepare dataset from text files
        prepare_dataset(
            args.input,
            args.output,
            args.tokenizer,
            num_workers=args.workers
        )
    
    elif args.task == 'split':
        # Split dataset into train/val/test
        split_dataset(args.input, args.output)
    
    elif args.task == 'shuffle':
        # Shuffle dataset
        shuffler = DataShuffler()
        shuffler.shuffle_file(args.input, args.output)


if __name__ == '__main__':
    main()