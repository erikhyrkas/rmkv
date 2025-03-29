# data/dataset.py

import os
import re
import pickle
import random
from typing import List, Dict, Any

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

        # Pre-compute attention masks for all samples
        self._precompute_attention_masks()

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

        # Store original length before truncation or padding
        original_length = len(token_ids)

        # Truncate if token_ids exceed max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            original_length = self.max_seq_len

        # Pad if token_ids are shorter than max_seq_len
        elif len(token_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_length

        # Return token_ids and original_length for mask calculation
        return {
            "token_ids": token_ids,
            "attention_length": original_length
        }

    def _precompute_attention_masks(self):
        """Pre-compute attention masks for all samples in a memory-efficient way."""
        print("Pre-computing attention masks...")

        # Initialize a list to store attention masks
        self.attention_masks = []

        # Convert samples to include attention masks
        processed_samples = []

        for sample_data in self.samples:
            token_ids = sample_data["token_ids"]
            attention_length = sample_data["attention_length"]

            # Create attention mask (-1e9 for padding, 0 for tokens) - OPTIMIZED
            attention_mask = [0] * attention_length
            if attention_length < self.max_seq_len:
                attention_mask.extend([-1e9] * (self.max_seq_len - attention_length))

            # Store the attention mask
            self.attention_masks.append(attention_mask)

            # Store only the token_ids in processed_samples to save memory
            processed_samples.append(token_ids)

        # Replace the samples with the token_ids only
        self.samples = processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.samples[idx]
        attention_mask = self.attention_masks[idx]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

class PretrainingDataset(Dataset):
    """
    A more efficient variant of the original PretrainingDataset that eliminates all padding
    by dynamically packing tokens across file boundaries.

    This implementation:
    1. Creates a continuous stream of tokens from all files
    2. Cuts this stream into exactly sized chunks of seq_len
    3. Preserves document boundaries to avoid learning across unrelated content
    4. Optionally shuffles the chunks for better training dynamics
    """

    def __init__(
            self,
            data_dir=None,
            seq_len=MODEL_CONFIG["max_seq_len"],
            cache_dir=None,
            doc_separator_token="<doc>",
            shuffle_chunks=True,
            use_attention_masks=False,
    ):
        # Basic initialization
        self.data_dir = data_dir or PATHS["data_dir"]
        self.seq_len = seq_len
        self.doc_separator_token = doc_separator_token
        self.shuffle_chunks = shuffle_chunks
        self.use_attention_masks = use_attention_masks

        # Set up cache directory - use a 'cache' subfolder in the data directory if not specified
        self.cache_dir = cache_dir or os.path.join(self.data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Path to the cached chunks file
        self.chunks_cache_path = os.path.join(self.cache_dir, 'packed_chunks.pkl')
        self.masks_cache_path = os.path.join(self.cache_dir, 'packed_masks.pkl')

        # Initialize the tokenizer
        self.tokenizer_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
        self.tokenizer = RemarkableTokenizer(load_path=self.tokenizer_path)

        # Get document separator token ID
        self.doc_sep_id = self.tokenizer.encode(self.doc_separator_token)[0]

        # List all text files in the directory
        self.file_paths = [
            os.path.join(self.data_dir, fname)
            for fname in os.listdir(self.data_dir)
            if (fname.endswith('.txt') or fname.endswith('.md')) and not fname.startswith('.')
        ]

        if not self.file_paths:
            raise ValueError(f"No training files found in directory {self.data_dir}.")

        # Load packed chunks and masks, using cache if available
        if self.use_attention_masks:
            self.chunks, self.attention_masks = self._load_chunks_with_cache()
        else:
            self.chunks, _ = self._load_chunks_with_cache()
            # Create a placeholder None for attention_masks
            self.attention_masks = None
        print(f"Loaded {len(self.chunks)} chunks for training.")

    def _load_chunks_with_cache(self):
        """Load packed chunks and masks, using cache if available and up-to-date."""
        print("Checking if chunks need to be rebuilt...")
        need_rebuild = (
                self._check_if_rebuild_needed(self.chunks_cache_path) or
                self._check_if_rebuild_needed(self.masks_cache_path)
        )

        if not need_rebuild and os.path.exists(self.chunks_cache_path) and os.path.exists(self.masks_cache_path):
            print("Loading packed chunks and masks from cache...")
            try:
                with open(self.chunks_cache_path, 'rb') as f:
                    chunks = pickle.load(f)
                with open(self.masks_cache_path, 'rb') as f:
                    masks = pickle.load(f)
                return chunks, masks
            except Exception as e:
                print(f"Error loading from cache: {e}. Rebuilding...")
                need_rebuild = True

        if need_rebuild:
            print("Building/rebuilding packed chunks and masks...")
            chunks, masks = self._create_packed_chunks()

            # Save the processed chunks and masks
            with open(self.chunks_cache_path, 'wb') as f:
                pickle.dump(chunks, f)
            with open(self.masks_cache_path, 'wb') as f:
                pickle.dump(masks, f)

            return chunks, masks

    def _check_if_rebuild_needed(self, cache_path):
        """Check if we need to rebuild the cache."""
        if not os.path.exists(cache_path):
            return True

        # Get the modification time of the cache file
        cache_mtime = os.path.getmtime(cache_path)

        # Check if tokenizer has been modified after the cache was created
        if os.path.getmtime(self.tokenizer_path) > cache_mtime:
            print("Tokenizer has changed. Rebuilding cache...")
            return True

        # Check if any source files have been modified
        for file_path in self.file_paths:
            if os.path.getmtime(file_path) > cache_mtime:
                print(f"File {os.path.basename(file_path)} has been modified. Rebuilding cache...")
                return True

        return False

    def _create_packed_chunks(self):
        """Create efficiently packed chunks and attention masks from all files."""
        # First, tokenize each file separately and keep track of document boundaries
        documents = []
        print("Creating packed chunks...")
        for file_path in self.file_paths:
            print(f"Tokenizing file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # For pretraining, remove instruction markers if present
            content = re.sub(r"<start>|<end>", "", content)

            # Tokenize the content
            doc_tokens = self.tokenizer.encode(content)
            documents.append(doc_tokens)

        # Create efficiently packed chunks and masks
        chunks, masks = self._pack_documents(documents)

        # Optionally shuffle chunks and masks together
        if self.shuffle_chunks:
            # Create an index list, shuffle it, and reorder the chunks and masks
            indices = list(range(len(chunks)))
            random.shuffle(indices)
            chunks = [chunks[i] for i in indices]
            masks = [masks[i] for i in indices]

        return chunks, masks

    def _pack_documents(self, documents: List[List[int]]) -> tuple:
        """
        Pack documents into fixed-size chunks efficiently.
        Makes sure no chunk contains a boundary in the middle.
        Returns both chunks and attention masks.
        """
        print("Packing documents...")
        chunks = []
        masks = []
        current_chunk = []

        for doc in documents:
            # If adding this document would exceed seq_len,
            # we need to handle the current chunk
            if len(current_chunk) + len(doc) + 1 > self.seq_len:
                # If we have a non-empty current chunk, add it to chunks
                if current_chunk:
                    # Create mask (1s for actual tokens, 0s for padding)
                    mask = [1] * len(current_chunk)

                    # Pad if necessary to reach seq_len
                    if len(current_chunk) < self.seq_len:
                        # Add padding to mask
                        mask.extend([0] * (self.seq_len - len(current_chunk)))
                        # Add padding to chunk
                        current_chunk.extend([self.tokenizer.pad_token_id] * (self.seq_len - len(current_chunk)))

                    chunks.append(current_chunk)
                    masks.append(mask)
                    current_chunk = []

                # Now we handle the new document
                # If it's too long for a single chunk, split it
                if len(doc) > self.seq_len:
                    for i in range(0, len(doc), self.seq_len):
                        doc_chunk = doc[i:min(i + self.seq_len, len(doc))]

                        if len(doc_chunk) == self.seq_len:
                            # Full chunk, add directly with an all-1s mask
                            chunks.append(doc_chunk)
                            masks.append([1] * self.seq_len)
                        else:
                            # Last piece of the document, start a new current_chunk
                            current_chunk = doc_chunk
                else:
                    # Document fits in a new chunk, so start with it
                    current_chunk = doc.copy()
            else:
                # We can add this document to the current chunk
                if current_chunk:
                    # Add document separator if not the first doc in the chunk
                    current_chunk.append(self.doc_sep_id)

                # Add the document
                current_chunk.extend(doc)

        # Don't forget the last chunk if it's not empty
        if current_chunk:
            # Create mask (1s for actual tokens, 0s for padding)
            mask = [1] * len(current_chunk)

            # Pad if necessary
            if len(current_chunk) < self.seq_len:
                # Add padding to mask
                mask.extend([0] * (self.seq_len - len(current_chunk)))
                # Add padding to chunk
                current_chunk.extend([self.tokenizer.pad_token_id] * (self.seq_len - len(current_chunk)))

            chunks.append(current_chunk)
            masks.append(mask)

        return chunks, masks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.use_attention_masks:
            return {
                "input_ids": torch.tensor(self.chunks[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.float)
            }
        else:
            return {
                "input_ids": torch.tensor(self.chunks[idx], dtype=torch.long)
            }