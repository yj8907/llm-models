

# from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import IterableDataset
import os
from typing import Iterator

import torch
from typing import Iterator
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset as HFDataset # Rename to avoid clash


# --- Configuration ---
MODEL_NAME = "gpt2"
CONTEXT_LENGTH = 1024
PROCESSED_DATA_DIR = "./processed_tinystories"

def tokenize_and_group_dataset(split: str):
    """Loads a split, tokenizes it, and groups it into fixed-size chunks."""
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Raw Dataset
    raw_dataset = load_dataset("roneneldan/TinyStories", split=split)

    # 3. Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False) # No truncation yet
    
    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=os.cpu_count(),
    )

    # 4. Grouping and Chunking
    def group_texts(examples):
        # Concatenate all lists of token IDs in the batch
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the last chunk
        total_length = (total_length // CONTEXT_LENGTH) * CONTEXT_LENGTH
        
        # Split the concatenated list into chunks of CONTEXT_LENGTH
        result = {
            k: [t[i : i + CONTEXT_LENGTH] for i in range(0, total_length, CONTEXT_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        # For CLM, labels are the input IDs
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )
    
    # 5. Save the processed dataset
    # output_path = os.path.join(PROCESSED_DATA_DIR, split)
    # lm_dataset.save_to_disk(output_path)
    # print(f"Processed dataset saved to: {output_path}")

    return lm_dataset


class TextDataset(Dataset):
    """
    Dataset for loading pre-tokenized text data.
    Expects data in the format of .bin files containing tokenized sequences.
    """
    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = torch.from_numpy(
            torch.load(data_path) if data_path.endswith('.pt') 
            else torch.ByteTensor(torch.ByteStorage.from_file(data_path, shared=False))
        ).long()
        
    def __len__(self):
        return len(self.data) // self.seq_len - 1
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = self.data[start_idx:end_idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class DolmaStreamingDataset(IterableDataset):
    """
    Streaming Dataset for loading pre-tokenized text data from Dolma.
    It uses shuffling built into the datasets library for efficient data mixing.
    """
    def __init__(self, data_path: str, seq_len: int, seed: int = 42):
        # Set the environment variable for data directory (if required by your setup)
        os.environ["DATA_DIR"] = data_path
        self.seq_len = seq_len
        self.seed = seed
        
        # Load the dataset in streaming mode
        self.data_stream = load_dataset("allenai/dolma", split="train", streaming=True)
        
    def _create_generator(self):
        """
        Creates an iterable that applies shuffling and then tokenization/sequence grouping.
        """
        # 1. Apply Shuffling
        # The .shuffle() method on a streaming dataset uses a buffer to efficiently 
        # mix samples, without loading the entire dataset into memory.
        shuffled_stream = self.data_stream.shuffle(seed=self.seed, buffer_size=10_000)
        
        # 2. Sequence Grouping / Tokenization (Conceptual Example)
        # In a real-world scenario, you would token-ID the text and then group
        # the resulting token sequences into chunks of size `seq_len`.
        
        # For demonstration, we'll iterate over the 'text' field and yield 
        # a sample until a more complex preprocessing pipeline is added.
        for item in shuffled_stream:
            text = item['text']
            # --- START: Placeholder for actual preprocessing logic ---
            # Assume 'text' is tokenized into a list of token IDs here.
            # Then, you'd yield (x, y) pairs of length seq_len.
            
            # Simple placeholder: just yield the text content
            yield text 
            # --- END: Placeholder for actual preprocessing logic ---

    def __iter__(self):
        # When PyTorch's DataLoader calls this, it gets the data generator
        # It ensures that each worker process (if using multiple) can create its own
        # independent stream.
        return self._create_generator()


class TinyStoriesTokenizedDataset(Dataset):
    """
    A PyTorch Dataset class that loads the *pre-processed*, tokenized, 
    and chunked TinyStories dataset from disk.
    
    It returns a dictionary suitable for a Hugging Face Trainer or custom PyTorch loop.
    """
    def __init__(self, split: str = "train", processed_data_dir: str = "./processed_tinystories"):
        """
        Initializes the dataset by loading the specified split from a local directory.

        Args:
            split (str): The dataset split to load ('train', 'validation').
            processed_data_dir (str): The directory where the pre-processed data is saved.
        """
        file_path = os.path.join(processed_data_dir, split)
        print(f"Loading *tokenized* TinyStories dataset split from: '{file_path}'...")
        
        # Load the pre-processed dataset from disk
        try:
            # We load the processed Arrow Table (Dataset)
            self.dataset = tokenize_and_group_dataset(split)
        except Exception as e:
            raise RuntimeError(
                f"Could not load processed dataset split '{split}'. "
                f"Did you run the pre-processing script first? Error: {e}"
            )

        self.split = split
        print(f"Tokenized dataset loaded. Total chunks: {len(self.dataset)}")

    def __len__(self):
        """
        Returns the total number of fixed-length chunks (examples) in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the token IDs and labels for the chunk at the given index.

        Args:
            idx (int): The index of the chunk to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids' and 'labels' tensors.
        """
        # Retrieve the example from the Hugging Face Dataset
        example = self.dataset[idx]
        
        # Convert the lists of integers (token IDs) into PyTorch Tensors
        # These are the necessary keys for Causal Language Modeling training
        return torch.tensor(example['input_ids'], dtype=torch.long), \
            torch.tensor(example['labels'], dtype=torch.long)

# Example Usage:
# train_dataset = TinyStoriesTokenizedDataset(split="train")
# val_dataset = TinyStoriesTokenizedDataset(split="validation")
# print(train_dataset[0])

