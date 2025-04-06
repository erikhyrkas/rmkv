# ./train_tokenizer_hf.py
#
# Trains a custom BPE tokenizer for the RMKV model using Hugging Face datasets.
# Combines data from multiple sources (Fineweb, reasoning, Nemotron) to create
# a balanced vocabulary that handles both general text and specialized content.
import os
import argparse

import torch
from datasets import load_dataset
from config import PATHS, MODEL_CONFIG
from data.tokenizer import RemarkableTokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import random


# python train_tokenizer_hf.py --vocab_size 30000 --max_samples 250000

def process_fineweb_text(tokenizer, text, max_length=1024):
    """
    Process text from the Fineweb dataset for tokenizer training.

    Args:
        tokenizer: Tokenizer instance for encoding/padding
        text (str): Raw text from Fineweb
        max_length (int): Target sequence length

    Returns:
        list or None: Token IDs of fixed length, or None if text is invalid

    Note:
        Handles padding for short sequences and truncation for long ones.
    """
    text = text.strip()
    if not text:
        return None
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens += [tokenizer.pad_token_id] * (max_length - len(tokens))
    return tokens


def process_reasoning_text(tokenizer, prompt, response, max_length=1024):
    """
    Process text from the reasoning dataset for tokenizer training.

    Formats the input with special tokens in the pattern:
    {prompt}<start><think>{response}</think><end>

    Args:
        tokenizer: Tokenizer instance for encoding/padding
        prompt (str): Instruction or question text
        response (str): Response or reasoning text
        max_length (int): Target sequence length

    Returns:
        list or None: Token IDs of fixed length, or None if text is invalid

    Note:
        The <think> tags wrap the reasoning portion to help the model
        learn to separate reasoning from final answers.
    """
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        return None
    text = f"{prompt}<start><think>{response}</think><end>"
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens += [tokenizer.pad_token_id] * (max_length - len(tokens))
    return tokens


def process_nemotron_text(tokenizer, prompt, response, max_length=1024):
    """
    Process text from the Nemotron dataset for tokenizer training.

    Extracts instruction and response from Nemotron's specific format
    with header tags, and formats them for the RMKV model.

    Args:
        tokenizer: Tokenizer instance for encoding/padding
        prompt (str): Raw prompt from Nemotron dataset
        response (str): Raw response from Nemotron dataset
        max_length (int): Target sequence length

    Returns:
        list or None: Token IDs of fixed length, or None if text is invalid

    Note:
        Handles Nemotron's specific format with <|end_header_id|> and <|eot_id|>
        markers, converting to RMKV's format with <start> and <end> tokens.
    """
    if "<|end_header_id|>" not in prompt:
        return None
    try:
        prompt_parts = prompt.split("user<|end_header_id|>")
        instruction_part = prompt_parts[1].split("<|eot_id|>")[0].strip()
        response_part = response.split("<|eot_id|>")[0].strip()
        full_text = f"{instruction_part}<start>{response_part}<end>"
        tokens = tokenizer.encode(full_text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif len(tokens) < max_length:
            tokens += [tokenizer.pad_token_id] * (max_length - len(tokens))
        return tokens
    except Exception:
        return None


class FinewebIterator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-51", split="train", streaming=True)
        self.iterator = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                row = next(self.iterator)
                text = row.get("text", "")
                tokens = process_fineweb_text(self.tokenizer, text, self.max_length)
                if tokens:
                    return {"input_ids": torch.tensor(tokens, dtype=torch.long)}
            except StopIteration:
                # Restart dataset iterator
                self.iterator = iter(self.dataset)
                continue
            except Exception:
                # Skip problematic entries
                continue


class ReasoningIterator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("glaiveai/reasoning-v1-20m", split="train")
        self.iterator = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                row = next(self.iterator)
                prompt = row.get("prompt", "")
                response = row.get("response", "")
                tokens = process_reasoning_text(self.tokenizer, prompt, response, self.max_length)
                if tokens:
                    return {"input_ids": torch.tensor(tokens, dtype=torch.long)}
            except StopIteration:
                # Restart dataset iterator
                self.iterator = iter(self.dataset)
                continue
            except Exception:
                # Skip problematic entries
                continue


class NemotronIterator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", "SFT",
                                    split=["code", "math", "science", "instruction following", "chat"],
                                    streaming=True)
        self.iterator = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                row = next(self.iterator)
                prompt = row.get("input", "")
                response = row.get("output", "")
                tokens = process_nemotron_text(self.tokenizer, prompt, response, self.max_length)
                if tokens:
                    return {"input_ids": torch.tensor(tokens, dtype=torch.long)}
            except StopIteration:
                # Restart dataset iterator
                self.iterator = iter(self.dataset)
                continue
            except Exception:
                # Skip problematic entries
                continue


def interleave_streams(tokenizer, max_length=1024, weights=(5, 2, 1)):
    """
    Creates an interleaved stream of data from multiple sources.
    Each source gets its own iterator with its own copy of the tokenizer to avoid
    the "Already mutably borrowed" error.
    """
    iterators = [
        FinewebIterator(tokenizer, max_length),
        ReasoningIterator(tokenizer, max_length),
        NemotronIterator(tokenizer, max_length)
    ]

    probs = [w / sum(weights) for w in weights]

    while True:
        i = random.choices(range(len(iterators)), weights=probs)[0]
        try:
            return next(iterators[i])
        except Exception as e:
            print(f"Error in stream {i}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer on Hugging Face datasets")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to stream (total from both datasets)")
    parser.add_argument("--fineweb_ratio", type=int, default=9, help="Ratio of fineweb samples")
    parser.add_argument("--reasoning_ratio", type=int, default=1, help="Ratio of reasoning samples")
    args = parser.parse_args()

    print("Training tokenizer with combined fineweb and reasoning data")
    print(f"Vocab size: {args.vocab_size}")
    if args.max_samples:
        print(f"Streaming up to {args.max_samples} total samples")

    tokenizer = RemarkableTokenizer(vocab_size=args.vocab_size)
    tokenizer.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Create an iterable class for training
    class TokenStreamIterable:
        """
        Creates a stream of text samples for tokenizer training.

        Combines data from multiple sources (Fineweb, reasoning, Nemotron)
        with weighted sampling to create a balanced training corpus
        for the tokenizer.

        Args:
            tokenizer: Tokenizer being trained
            max_length (int): Maximum sequence length
            weights (tuple): Relative sampling weights for each data source
            max_samples (int): Maximum number of samples to provide

        Yields:
            str: Text samples suitable for tokenizer training

        Note:
            This class handles all the complexity of loading and interleaving
            multiple datasets, with error handling to ensure training proceeds
            even if individual samples or datasets fail.
        """
        def __init__(self, tokenizer, max_length, weights, max_samples):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.weights = weights
            self.max_samples = max_samples
            self.count = 0

            # Load datasets once
            print("Initializing datasets...")
            self.fineweb_dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-51",
                                                split="train", streaming=True)
            self.fineweb_iter = iter(self.fineweb_dataset)

            self.reasoning_dataset = load_dataset("glaiveai/reasoning-v1-20m", split="train")
            self.reasoning_size = len(self.reasoning_dataset)

            # Load nemotron datasets
            self.nemotron_datasets = {}
            self.nemotron_iters = {}
            for dataset_name in ['science', 'math', 'chat', 'code']:
                try:
                    self.nemotron_datasets[dataset_name] = load_dataset(
                        "nvidia/Llama-Nemotron-Post-Training-Dataset-v1",
                        "SFT", split=dataset_name, streaming=True)
                    self.nemotron_iters[dataset_name] = iter(self.nemotron_datasets[dataset_name])
                except Exception as e:
                    print(f"Error loading {dataset_name} dataset: {e}")
            print("Datasets initialized successfully.")

        def __iter__(self):
            return self

        def get_fineweb_sample(self):
            """Get a sample from the Fineweb dataset"""
            max_tries = 10  # Avoid infinite loops
            for _ in range(max_tries):
                try:
                    row = next(self.fineweb_iter)
                    text = row.get("text", "").strip()
                    if text:
                        return text
                except StopIteration:
                    # Restart iterator if we reach the end
                    self.fineweb_iter = iter(self.fineweb_dataset)
                except Exception:
                    # Skip problematic entries
                    pass
            return None  # Return None if max_tries exceeded

        def get_reasoning_sample(self):
            """Get a sample from the Reasoning dataset"""
            max_tries = 10  # Avoid infinite loops
            for _ in range(max_tries):
                try:
                    idx = random.randint(0, self.reasoning_size - 1)
                    row = self.reasoning_dataset[idx]
                    prompt = row.get("prompt", "").strip()
                    response = row.get("response", "").strip()
                    if prompt and response:
                        return f"{prompt}<start><think>{response}</think><end>"
                except Exception:
                    # Skip problematic entries
                    pass
            return None

        def get_nemotron_sample(self):
            """Get a sample from a Nemotron dataset"""
            # Randomly select one of the available datasets
            dataset_names = list(self.nemotron_iters.keys())
            if not dataset_names:
                return None

            # Try each dataset in random order
            random.shuffle(dataset_names)
            for dataset_name in dataset_names:
                max_tries = 5  # Limit tries per dataset
                for _ in range(max_tries):
                    try:
                        row = next(self.nemotron_iters[dataset_name])
                        prompt = row.get("input", "").strip()
                        response = row.get("output", "").strip()
                        if "<|end_header_id|>" in prompt:
                            prompt_parts = prompt.split("user<|end_header_id|>")
                            instruction_part = prompt_parts[1].split("<|eot_id|>")[0].strip()
                            response_part = response.split("<|eot_id|>")[0].strip()
                            return f"{instruction_part}<start>{response_part}<end>"
                    except StopIteration:
                        # Restart iterator if we reach the end
                        self.nemotron_iters[dataset_name] = iter(self.nemotron_datasets[dataset_name])
                    except Exception:
                        # Skip problematic entries
                        pass
            return None

        def __next__(self):
            if self.max_samples and self.count >= self.max_samples:
                raise StopIteration

            self.count += 1
            i = random.choices(range(len(self.weights)), weights=self.weights)[0]

            max_attempts = 5  # Maximum number of attempts to get a valid sample
            for _ in range(max_attempts):
                if i == 0:  # Fineweb
                    sample = self.get_fineweb_sample()
                elif i == 1:  # Reasoning
                    sample = self.get_reasoning_sample()
                else:  # Nemotron
                    sample = self.get_nemotron_sample()

                if sample:
                    return sample

                # If we didn't get a sample, try another source
                i = (i + 1) % len(self.weights)

            # If we still don't have a sample after trying all sources, return a simple fallback
            return "Fallback text for tokenizer training."
    trainer = BpeTrainer(
        vocab_size=30000,
        special_tokens=["<think>", "</think>", "<start>", "<end>", "<pad>", "<unk>"],
        show_progress=True
    )
    print("BpeTrainer instantiated successfully.")

    print("Starting tokenizer training...")
    # Create a text stream that doesn't use the tokenizer internally
    # This avoids the conflict of training a tokenizer while using it
    text_stream = TokenStreamIterable(
        tokenizer,
        max_length=MODEL_CONFIG["max_seq_len"],
        weights=[5, 1, 1],
        max_samples=args.max_samples
    )

    # Train from text iterator
    # This approach allows streaming training without loading all data into memory
    tokenizer.tokenizer.train_from_iterator(text_stream, trainer=trainer)

    save_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    main()
