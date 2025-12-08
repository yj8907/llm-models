

from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader


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


class TinyStoriesDataset(Dataset):
    """
    A PyTorch Dataset class for loading the TinyStories dataset.

    This class loads the entire specified split into memory (as an Arrow table)
    allowing for standard random access and efficient shuffling by the DataLoader.
    """
    def __init__(self, split: str = "train"):
        """
        Initializes the dataset by loading the specified split from Hugging Face.

        Args:
            split (str): The dataset split to load ('train', 'validation', or 'test').
        """
        print(f"Loading TinyStories dataset split: '{split}'...")
        
        # Load the specified split. The dataset is automatically downloaded and cached.
        try:
            self.data: HFDataset = load_dataset("roneneldan/TinyStories", split=split)
        except Exception as e:
            raise RuntimeError(f"Could not load TinyStories split '{split}'. Error: {e}")

        self.split = split
        print(f"Dataset loaded. Total stories: {len(self.data)}")

    def __len__(self):
        """
        Returns the total number of stories in the dataset split.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """
        Retrieves the story text at the given index.

        Args:
            idx (int): The index of the story to retrieve.

        Returns:
            str: The text content of the story.
        """
        # The 'text' column holds the story content
        story_text = self.data[idx]['text']
        
        # In a real training pipeline, you would typically return token IDs here
        # E.g., token_ids = self.tokenizer.encode(story_text)
        # return torch.tensor(token_ids, dtype=torch.long)
        
        return story_text

