# data/dataset.py

import os
import re

import torch
from torch.utils.data import Dataset
from config import PATHS, MODEL_CONFIG
from data.tokenizer import RemarkableTokenizer


class InstructionDataset(Dataset):
    """
    A dataset for instruction-tuning that loads training data from a directory containing many text files.
    Each file is expected to contain multiple instructions formatted as:

    <prompt><start><response><end>

    For example:
        What is a frog?<start>An amphibian.<end>

    The dataset extracts each instruction, combines the prompt and response into a single text sample,
    and tokenizes the result.
    """

    def __init__(self, data_dir=None, max_seq_len=MODEL_CONFIG["max_seq_len"]):
        # Use the provided directory or default from PATHS
        self.data_dir = data_dir or PATHS["data_dir"]
        self.max_seq_len = max_seq_len
        self.tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))

        # List all text files in the directory; assumes files end with '.txt'/'.md'
        self.file_paths = [
            os.path.join(self.data_dir, fname)
            for fname in os.listdir(self.data_dir)
            if fname.endswith('.txt') or fname.endswith('.md')
        ]

        if not self.file_paths:
            raise ValueError(f"No training files found in directory {self.data_dir}.")

        # Load all instruction samples from the files
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError("No valid instructions found in the training files. Check the file format.")

    def _load_samples(self):
        samples = []
        # Pattern to capture prompt and response. Dot matches newline with re.DOTALL.
        pattern = re.compile(r"(.*?)<start>(.*?)<end>", re.DOTALL)
        for file_path in self.file_paths:
            print(f"Loading data from: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Find all matches in the file
            instructions = pattern.findall(content)
            # Each match is a tuple: (prompt, response)
            for prompt, response in instructions:
                # Combine prompt and response with explicit delimiters
                full_text = prompt.strip() + " <start> " + response.strip() + " <end>"
                if full_text:
                    samples.append(self.tokenize(full_text))
        print(f"Loaded {len(samples)} samples.")
        return samples

    def tokenize(self, text):
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        # Truncate if token_ids exceed max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        # Pad if token_ids are shorter than max_seq_len (assuming pad token id is defined, e.g., 0)
        elif len(token_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_length
        return token_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.samples[idx]
        sample = {"input_ids": torch.tensor(token_ids, dtype=torch.long)}
        return sample


if __name__ == "__main__":
    dataset = InstructionDataset()
    print(f"Loaded {len(dataset)} instruction samples from {len(dataset.file_paths)} files.")
    # Print tokenized version of the first sample.
    print(dataset[0])
