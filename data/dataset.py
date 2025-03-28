# data/dataset.py

import os
import re
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset
from config import PATHS, MODEL_CONFIG
from data.tokenizer import RemarkableTokenizer


class InstructionDataset(Dataset):
    """
    A dataset for instruction-tuning that loads training data from a directory containing many text files.
    Each file is expected to contain multiple instructions formatted as:

    [prompt text]<start>[<think>optional chain of thought</think>][response text]<end>

    For example:
        What is a frog?<start>An amphibian.<end>

    The dataset extracts each instruction, combines the prompt and response into a single text sample,
    and tokenizes the result. It supports caching to avoid reprocessing unchanged files.
    """

    def __init__(self, data_dir=None, max_seq_len=MODEL_CONFIG["max_seq_len"], cache_dir=None):
        # Use the provided directory or default from PATHS
        self.data_dir = data_dir or PATHS["data_dir"]
        self.max_seq_len = max_seq_len

        # Set up cache directory - use a 'cache' subfolder in the data directory if not specified
        self.cache_dir = cache_dir or os.path.join(self.data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Path to the cached samples file
        self.samples_cache_path = os.path.join(self.cache_dir, 'samples.pkl')

        # Initialize the tokenizer
        self.tokenizer_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
        self.tokenizer = RemarkableTokenizer(load_path=self.tokenizer_path)

        # List all text files in the directory; assumes files end with '.txt'/'.md'
        self.file_paths = [
            os.path.join(self.data_dir, fname)
            for fname in os.listdir(self.data_dir)
            if (fname.endswith('.txt') or fname.endswith('.md')) and not fname.startswith('.')
        ]

        if not self.file_paths:
            raise ValueError(f"No training files found in directory {self.data_dir}.")

        # Load all instruction samples from the files, using cache if available
        self.samples = self._load_samples_with_cache()
        if not self.samples:
            raise ValueError("No valid instructions found in the training files. Check the file format.")

    def _load_samples_with_cache(self):
        """Load samples, using cache if available and up-to-date."""
        # Check if we need to rebuild the cache
        need_rebuild = self._check_if_rebuild_needed()

        if not need_rebuild and os.path.exists(self.samples_cache_path):
            print("Loading samples from cache...")
            try:
                with open(self.samples_cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading from cache: {e}. Rebuilding...")
                need_rebuild = True

        if need_rebuild:
            print("Building/rebuilding cache...")
            samples = self._load_and_process_samples()

            # Save the processed samples
            with open(self.samples_cache_path, 'wb') as f:
                pickle.dump(samples, f)

            return samples

    def _check_if_rebuild_needed(self):
        """Check if we need to rebuild the cache based on file modification times."""
        if not os.path.exists(self.samples_cache_path):
            return True

        # Get the modification time of the cache file
        cache_mtime = os.path.getmtime(self.samples_cache_path)

        # Check if tokenizer has been modified after the cache was created
        if os.path.getmtime(self.tokenizer_path) > cache_mtime:
            print("Tokenizer has changed. Rebuilding cache...")
            return True

        # Check if any source files have been modified after the cache was created
        for file_path in self.file_paths:
            if os.path.getmtime(file_path) > cache_mtime:
                print(f"File {os.path.basename(file_path)} has been modified. Rebuilding cache...")
                return True

        return False

    def _load_and_process_samples(self):
        """Load and process all samples from scratch."""
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