# ./train_hf.py
#
# Training script for the RMKV model using Hugging Face datasets.
# Supports two training modes:
# 1. Pretrain: Initial training on web text (Fineweb)
# 2. Finetune: Instruction tuning on reasoning and instruction datasets
#
# The script implements interleaved streaming from multiple data sources
# to balance different data types during training.
import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from config import MODEL_CONFIG, PRETRAIN_CONFIG, FINETUNE_CONFIG, PATHS
from model.rmkv import RMKVModel
from data.tokenizer import RemarkableTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import time
import re
import random

# this is a little bit of a mess, really need a shared utility file...
# Import the needed functions from train_tokenizer_hf
from train_tokenizer_hf import process_fineweb_text, process_reasoning_text, process_nemotron_text
from training.checkpoint import save_checkpoint


# python train_hf.py --mode pretrain
# python train_hf.py --mode finetune

# -------------------------
# Stream Dataset Iterators
# -------------------------
class FinewebIterator:
    """
    Iterator for streaming data from the Hugging Face Fineweb dataset.

    Provides an infinite stream of text samples from web content,
    suitable for pretraining language models on diverse content.

    Args:
        tokenizer: Tokenizer for encoding text
        max_length (int): Maximum sequence length for tokenized samples

    Yields:
        dict: Contains "input_ids" with tokenized text of fixed length

    Note:
        Handles exceptions gracefully to ensure the training loop
        never crashes due to problematic samples.
    """

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
    """
    Iterator for streaming data from the reasoning dataset.

    Provides prompt-response pairs formatted with special tokens
    for instruction tuning with reasoning capabilities.

    Args:
        tokenizer: Tokenizer for encoding text
        max_length (int): Maximum sequence length for tokenized samples

    Yields:
        dict: Contains "input_ids" with tokenized prompt-response pairs

    Note:
        Formats data with <start>, <think>, and <end> special tokens to
        support the reasoning pattern in the model's training.
    """

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
    """
    Iterator for streaming data from Nvidia's Nemotron instruction dataset.

    Provides instruction-response pairs across multiple domains (code, math,
    science, instruction following, chat) for balanced instruction tuning.

    Args:
        tokenizer: Tokenizer for encoding text
        max_length (int): Maximum sequence length for tokenized samples

    Yields:
        dict: Contains "input_ids" with tokenized instruction-response pairs

    Note:
        Dynamically rotates between dataset splits to ensure diversity,
        handling any loading errors or stream interruptions gracefully.
    """

    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Load just one split at a time to avoid the list issue
        self.datasets = {}
        self.iterators = {}
        self.current_split_idx = 0
        self.splits = ["code", "math", "science", "instruction following", "chat"]

        # Initialize with the first split
        self.init_current_split()

    def init_current_split(self):
        """Initialize the current split"""
        current_split = self.splits[self.current_split_idx]
        if current_split not in self.datasets:
            try:
                self.datasets[current_split] = load_dataset(
                    "nvidia/Llama-Nemotron-Post-Training-Dataset-v1",
                    "SFT",
                    split=current_split,
                    streaming=True
                )
                self.iterators[current_split] = iter(self.datasets[current_split])
            except Exception as e:
                print(f"Error loading {current_split} dataset: {e}")
                # Move to next split on error
                self.current_split_idx = (self.current_split_idx + 1) % len(self.splits)
                self.init_current_split()

    def __iter__(self):
        return self

    def __next__(self):
        current_split = self.splits[self.current_split_idx]
        max_tries = 3

        for _ in range(max_tries * len(self.splits)):  # Try all splits if needed
            try:
                row = next(self.iterators[current_split])
                prompt = row.get("input", "")
                response = row.get("output", "")
                tokens = process_nemotron_text(self.tokenizer, prompt, response, self.max_length)
                if tokens:
                    return {"input_ids": torch.tensor(tokens, dtype=torch.long)}
            except StopIteration:
                # Rotate to next split
                self.current_split_idx = (self.current_split_idx + 1) % len(self.splits)
                current_split = self.splits[self.current_split_idx]
                # Initialize if needed
                if current_split not in self.iterators:
                    self.init_current_split()
                else:
                    # Restart the iterator
                    self.iterators[current_split] = iter(self.datasets[current_split])
            except Exception as e:
                # Skip problematic entries
                continue

        # If we're here, we failed to get a valid sample from any split
        # Return a dummy sample instead of raising an exception
        dummy_tokens = [self.tokenizer.pad_token_id] * self.max_length
        return {"input_ids": torch.tensor(dummy_tokens, dtype=torch.long)}


# -------------------------
# Stream Dataset Wrapper
# -------------------------
class InterleaveDataset(IterableDataset):
    """
    An IterableDataset that interleaves samples from multiple data streams.

    Combines data from different sources (Fineweb, Reasoning, Nemotron) with
    weighted sampling to control the ratio of different data types during training.

    Args:
        tokenizer: Tokenizer for encoding text
        max_length (int): Maximum sequence length for tokenized samples
        weights (tuple): Relative sampling weights for each dataset stream
                        (Fineweb, Reasoning, Nemotron)

    Yields:
        dict: Contains "input_ids" with tokenized text from a randomly
             selected data source based on specified weights

    Note:
        This approach allows mixing different data types during training without
        needing to create a static combined dataset, saving disk space and
        enabling infinite streaming for long training runs.
    """

    def __init__(self, tokenizer, max_length=1024, weights=(5, 1, 1)):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = weights
        self.iterators = [
            FinewebIterator(tokenizer, max_length),
            ReasoningIterator(tokenizer, max_length),
            NemotronIterator(tokenizer, max_length)
        ]
        self.probs = [w / sum(weights) for w in weights]

    def __iter__(self):
        while True:
            i = random.choices(range(len(self.iterators)), weights=self.probs)[0]
            try:
                yield next(self.iterators[i])
            except Exception as e:
                print(f"Error in stream {i}: {e}")
                continue


# -------------------------
# Hugging Face Dataset Wrappers
# -------------------------
class HFStreamingDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, max_len=1024):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        for sample in self.dataset:
            text = sample.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            elif len(tokens) < self.max_len:
                tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            yield {
                "input_ids": torch.tensor(tokens, dtype=torch.long)
            }


class HFFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=1024):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer
        for sample in hf_dataset:
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            text = f"{prompt.strip()}<start>{response.strip()}<end>"
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_len:
                tokens += [self.tokenizer.pad_token_id] * (max_len - len(tokens))
                self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx]}


# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model, tokenizer, device, mode, max_samples=50):
    """
    Evaluate model performance on held-out samples.

    Calculates loss on a sample of data to track training progress.
    Different evaluation datasets are used depending on training mode.

    Args:
       model: The RMKV model to evaluate
       tokenizer: Tokenizer for encoding text
       device: Computation device (CPU/GPU)
       mode (str): "pretrain" or "finetune" - determines evaluation data
       max_samples (int): Number of samples to evaluate on

    Returns:
       None, but prints average loss to console

    Note:
       For pretrain mode, evaluates on Fineweb samples
       For finetune mode, evaluates on reasoning dataset samples
    """
    model.eval()
    with torch.no_grad():
        if mode == "pretrain":
            dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-51", split="train", streaming=True)
            stream = (s for s in dataset if s.get("text"))
            texts = [next(stream)["text"] for _ in range(max_samples)]
        else:
            dataset = load_dataset("glaiveai/reasoning-v1-20m", split="train")
            texts = [f"{s['prompt']}<start>{s['response']}<end>" for s in dataset.select(range(max_samples))]

        total_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > MODEL_CONFIG["max_seq_len"]:
                tokens = tokens[:MODEL_CONFIG["max_seq_len"]]
            else:
                tokens += [tokenizer.pad_token_id] * (MODEL_CONFIG["max_seq_len"] - len(tokens))

            # Model now handles tensor conversion internally
            outputs = model(tokens)

            # But we still need to ensure input_tensor for the labels
            input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            labels = input_tensor.clone()

            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / max_samples
        print(f"\n[Eval] Avg loss on {max_samples} samples: {avg_loss:.4f}\n")
    model.train()


# -------------------------
# Training Function
# -------------------------
def train(model, dataloader, optimizer, scheduler, device, config, pad_token_id, tokenizer, mode, start_step=0):
    """
    Main training loop for the RMKV model.

    Handles both pretraining and finetuning with appropriate configurations,
    including gradient accumulation, logging, and checkpointing.

    Args:
        model: The RMKV model to train
        dataloader: DataLoader providing training batches
        optimizer: Optimizer instance (typically AdamW)
        scheduler: Learning rate scheduler
        device: Computation device (CPU/GPU)
        config (dict): Training configuration parameters
        pad_token_id (int): Token ID for padding (ignored in loss calculation)
        tokenizer: Tokenizer instance
        mode (str): "pretrain" or "finetune"
        start_step (int): Step to resume training from

    Process:
        1. Sets model to training mode
        2. Iterates through batches with progress display
        3. Computes loss and updates model with gradient accumulation
        4. Logs progress periodically and saves checkpoints
        5. Ends when max_steps is reached or all epochs complete
    """
    model.train()
    step = start_step
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"{mode} Epoch {epoch + 1}", dynamic_ncols=True)
        total_loss = 0
        for batch in pbar:
            if config.get("max_steps") and step >= config["max_steps"]:
                elapsed = time.time() - start_time
                summary_path = os.path.join(PATHS["logs_dir"], f"{mode}_summary.log")
                with open(summary_path, "w") as f:
                    f.write(f"Final loss: {total_loss / (step + 1):.4f}\n")
                    f.write(f"Total steps: {step}\n")
                    f.write(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}\n")
                print(f"Training complete. Summary written to {summary_path}")
                return

            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()

            # Model handles tensor shape internally now
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            steps_remaining = config.get("max_steps", 0) - step if config.get("max_steps") else 0
            eta = steps_remaining / (step / elapsed) if step > 0 and config.get("max_steps") else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta)) if eta else "--"

            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "ETA": eta_str})

            if step % 1000 == 0 and step > 0:
                with open(os.path.join(PATHS["logs_dir"], f"{mode}_step{step}.log"), "w") as f:
                    f.write(f"Step {step} loss: {avg_loss:.4f}\n")

            if step % 5000 == 0 and step > 0:
                ckpt_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{mode}_step{step}.pt")
                save_checkpoint(model, optimizer, epoch=epoch, global_step=step, filepath=ckpt_path)

            step += 1


# -------------------------
# Main Entry Point
# -------------------------
def main(mode="pretrain"):
    """
    Entry point for training the RMKV model.

    Sets up the complete training pipeline:
    - Loads model and tokenizer
    - Configures dataset streams
    - Creates optimizer and scheduler
    - Initializes from checkpoints if available
    - Runs the training loop
    - Saves the final model

    Args:
        mode (str): "pretrain" or "finetune" - determines training configuration

    Notes:
        - Pretraining focuses on general language modeling from web text
        - Finetuning focuses on instruction-following and reasoning capabilities
        - Learning rates, batch sizes, and dataset mixtures differ by mode
        - Automatically resumes from the latest checkpoint if available
    """
    assert mode in ["pretrain", "finetune"], "Invalid mode"
    config = PRETRAIN_CONFIG if mode == "pretrain" else FINETUNE_CONFIG
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))
    model = RMKVModel(tokenizer.vocab_size_actual).to(device)

    # Look for existing checkpoints to resume from
    start_step = 0
    ckpts = [f for f in os.listdir(PATHS["checkpoint_dir"]) if f.startswith(f"rmkv_{mode}_step") and f.endswith(".pt")]
    if ckpts:
        # Sort by step number to find the latest checkpoint
        ckpts.sort(key=lambda f: int(re.findall(r'step(\d+)', f)[0]))
        latest_ckpt = ckpts[-1]
        checkpoint_path = os.path.join(PATHS["checkpoint_dir"], latest_ckpt)
        if load_from_checkpoint(checkpoint_path, model, device):
            start_step = int(re.findall(r'step(\d+)', latest_ckpt)[0])
            print(f"[Resume] Loaded checkpoint: {latest_ckpt} at step {start_step}")
        else:
            print(f"[Warning] Failed to load checkpoint: {latest_ckpt}. Starting from scratch.")

    if mode == "pretrain":
        # Use our proper IterableDataset implementation for pretraining
        # Weights: (5, 1, 1) prioritizes Fineweb data (web text) for general knowledge
        dataset = InterleaveDataset(tokenizer, MODEL_CONFIG["max_seq_len"], weights=(5, 1, 1))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    else:
        # SFT mode (Supervised Fine-Tuning) uses different data mixture
        # Weights: (0, 2, 1) focuses on reasoning and instruction data only
        dataset = InterleaveDataset(tokenizer, MODEL_CONFIG["max_seq_len"], weights=(0, 2, 1))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Calculate total training steps based on data size or max_steps config
    if config.get("max_steps"):
        num_training_steps = config["max_steps"]
    else:
        # Estimate steps based on approximate dataset sizes and batch size
        if mode == "pretrain":
            # Approximate size of Fineweb dataset / (sequence_length * batch_size)
            approx_steps_per_epoch = int(131200000000 / (MODEL_CONFIG["max_seq_len"] * config["batch_size"]))
        else:
            # Approximate size of reasoning dataset / (sequence_length * batch_size)
            approx_steps_per_epoch = int(22000000 / (MODEL_CONFIG["max_seq_len"] * config["batch_size"]))
        num_training_steps = approx_steps_per_epoch * config["num_epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps
    )

    train(model, dataloader, optimizer, scheduler, device, config, tokenizer.pad_token_id, tokenizer, mode,
          start_step=start_step)

    save_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{mode}_final.pt")
    save_checkpoint(model, optimizer, epoch=0, global_step=0, filepath=save_path)

    # Also save as latest.pt for easy reference in run.py
    latest_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
    save_checkpoint(model, optimizer, epoch=0, global_step=0, filepath=latest_path)

    print(f"Model saved to {save_path} and {latest_path}")


if __name__ == "__main__":
    import argparse
    from training.checkpoint import load_from_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune"], default="pretrain")
    args = parser.parse_args()
    main(args.mode)
