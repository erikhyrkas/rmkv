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
from training.checkpoint import save_checkpoint


# python train_hf.py --mode pretrain
# python train_hf.py --mode finetune


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

    # Create attention mask (0 for real tokens, -1e9 for padding)
    attention_mask = [0] * min(len(tokens), max_length)

    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        padding_length = max_length - len(tokens)
        # Add padding to tokens
        tokens += [tokenizer.pad_token_id] * padding_length
        # Add mask values for padding
        attention_mask += [-1e9] * padding_length

    return {"input_ids": tokens, "attention_mask": attention_mask}


def process_reasoning_text(tokenizer, prompt, response, for_pretraining, max_length=1024):
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

    if for_pretraining:
        text = prompt + " " + response
        text = re.sub(r"<think>|</think>", " ", text).strip()
    else:
        text = f"{prompt}<start>{response}<end>"

    tokens = tokenizer.encode(text)

    # Create attention mask (0 for real tokens, -1e9 for padding)
    attention_mask = [0] * min(len(tokens), max_length)

    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        padding_length = max_length - len(tokens)
        tokens += [tokenizer.pad_token_id] * padding_length
        attention_mask += [-1e9] * padding_length

    return {"input_ids": tokens, "attention_mask": attention_mask}


def process_nemotron_text(tokenizer, prompt, response, for_pretraining, max_length=1024):
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
        if for_pretraining:
            full_text = instruction_part + " " + response_part
            full_text = re.sub(r"<think>|</think>", " ", full_text).strip()
        else:
            full_text = f"{instruction_part}<start>{response_part}<end>"
        tokens = tokenizer.encode(full_text)

        # Create attention mask (0 for real tokens, -1e9 for padding)
        attention_mask = [0] * min(len(tokens), max_length)

        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            padding_length = max_length - len(tokens)
            tokens += [tokenizer.pad_token_id] * padding_length
            attention_mask += [-1e9] * padding_length

        return {"input_ids": tokens, "attention_mask": attention_mask}
    except Exception:
        return None


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
                processed = process_fineweb_text(self.tokenizer, text, self.max_length)
                if processed:
                    return {
                        "input_ids": torch.tensor(processed["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(processed["attention_mask"], dtype=torch.float)
                    }
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

    def __init__(self, tokenizer, for_pretraining, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("glaiveai/reasoning-v1-20m", split="train")
        self.iterator = iter(self.dataset)
        self.for_pretraining = for_pretraining

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
                    return {
                        "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.float)
                    }
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

    def __init__(self, tokenizer, for_pretraining, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Load just one split at a time to avoid the list issue
        self.datasets = {}
        self.iterators = {}
        self.current_split_idx = 0
        self.splits = ["code", "math", "science", "instruction following", "chat"]
        self.for_pretraining = for_pretraining

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
                tokens = process_nemotron_text(self.tokenizer, prompt, response, self.for_pretraining, self.max_length)
                if tokens:
                    return {
                        "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.float)
                    }
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

    def __init__(self, tokenizer, for_pretraining, max_length=1024, weights=(5, 1, 1)):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = weights
        self.for_pretraining = for_pretraining
        self.iterators = [
            FinewebIterator(tokenizer, max_length),
            ReasoningIterator(tokenizer, self.for_pretraining, max_length),
            NemotronIterator(tokenizer, self.for_pretraining, max_length)
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
            # Create attention mask
            attention_mask = [0] * min(len(tokens), MODEL_CONFIG["max_seq_len"])

            if len(tokens) > MODEL_CONFIG["max_seq_len"]:
                tokens = tokens[:MODEL_CONFIG["max_seq_len"]]
            else:
                padding_length = MODEL_CONFIG["max_seq_len"] - len(tokens)
                attention_mask += [-1e9] * padding_length
                tokens += [tokenizer.pad_token_id] * padding_length

            input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            labels = input_tensor.clone()
            attention_tensor = torch.tensor(attention_mask, dtype=torch.float).unsqueeze(0).to(device)

            # Model now handles tensor conversion internally
            outputs = model(input_tensor, attention_tensor)

            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

        avg_loss = total_loss / max_samples
        print(f"\n[Eval] Avg loss on {max_samples} samples: {avg_loss:.4f}\n")
    model.train()


# -------------------------
# Training Function
# -------------------------
def train(model: RMKVModel, dataloader, optimizer, scheduler, device, config, pad_token_id, tokenizer, mode, start_step=0):
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
    is_flow = mode == "flow"

    for epoch in range(config["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"{mode} Epoch {epoch + 1}", dynamic_ncols=True)
        total_loss = 0

        for batch in pbar:
            if config.get("max_steps") and step >= config["max_steps"]:
                elapsed = time.time() - start_time
                summary_path = os.path.join(PATHS["logs_dir"], f"{mode}_summary.log")
                with open(summary_path, "w") as f:
                    f.write(f"Final loss: {total_loss / (step + 1):.4f}\\n")
                    f.write(f"Total steps: {step}\\n")
                    f.write(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}\\n")
                print(f"Training complete. Summary written to {summary_path}")
                return

            if is_flow:
                input_ids = batch["input_ids"].to(device)  # [B, S, L]
                attention_mask = batch["attention_mask"].to(device)  # [B, S, L]
                B, S, L = input_ids.shape
                memory = model.initial_memory.expand(B, -1, -1).to(device)
                batch_loss = 0

                for i in range(S):
                    segment = input_ids[:, i]  # [B, L]
                    mask = attention_mask[:, i]  # [B, L]

                    inputs = segment[:, :-1]
                    labels = segment[:, 1:]
                    input_mask = mask[:, :-1]

                    # 1016.0 real tokens in batch
                    # if step % 100 == 0:
                    #     num_valid = input_mask.sum().item()
                    #     print(f"[Debug] Step {step}: {num_valid} real tokens in batch")

                    logits, memory = model.generate_step(inputs.tolist(), memory, input_mask)

                    active_loss = input_mask.reshape(-1) > 0
                    active_logits = logits.reshape(-1, logits.size(-1))[active_loss]
                    active_labels = labels.reshape(-1)[active_loss]
                    loss = loss_fn(active_logits, active_labels)
                    batch_loss += loss

                loss = batch_loss / S


            else:
                inputs = batch["input_ids"].to(device)
                labels = inputs.clone()
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                outputs = model(inputs, attention_mask=attention_mask)
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
                    f.write(f"Step {step} loss: {avg_loss:.4f}\\n")

            if step % 5000 == 0 and step > 0:
                ckpt_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{mode}_step{step}.pt")
                save_checkpoint(model, optimizer, epoch=epoch, global_step=step, filepath=ckpt_path)

            step += 1


class SegmentedDataset(IterableDataset):
    def __init__(self, tokenizer, source="fineweb", segment_len=128, segments_per_sample=8):
        self.tokenizer = tokenizer
        self.segment_len = segment_len
        self.segments_per_sample = segments_per_sample
        self.source = source

        if source == "fineweb":
            self.stream = iter(load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-51", split="train", streaming=True))
        elif source == "reasoning":
            self.stream = iter(load_dataset("glaiveai/reasoning-v1-20m", split="train"))
        else:
            raise ValueError("Unsupported flow source")

    def __iter__(self):
        return self

    def __next__(self):
        tokens = []
        while len(tokens) < self.segment_len * self.segments_per_sample:
            try:
                row = next(self.stream)
                text = row.get("text", row.get("prompt", "") + " " + row.get("response", ""))
                t = self.tokenizer.encode(text.strip())
                tokens.extend(t)
            except Exception:
                continue  # skip bad rows

        chunks = [tokens[i:i + self.segment_len] for i in range(0, len(tokens), self.segment_len)]
        padded_chunks = []
        attention_masks = []

        for chunk in chunks[:self.segments_per_sample]:
            padding_len = self.segment_len - len(chunk)
            if padding_len > 0:
                padded_chunk = chunk + [self.tokenizer.pad_token_id] * padding_len
                mask = [1.0] * len(chunk) + [0.0] * padding_len
            else:
                padded_chunk = chunk[:self.segment_len]
                mask = [1.0] * self.segment_len

            padded_chunks.append(padded_chunk)
            attention_masks.append(mask)

        input_tensor = torch.tensor(padded_chunks, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.float)
        return {"input_ids": input_tensor, "attention_mask": attention_mask}


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
        mode (str): "focus", "flow", or "finetune" - determines training configuration

    Notes:
        - focus: full sequence training without memory reuse
        - flow: segmented training with memory reuse across segments
        - finetune: instruction tuning on curated datasets
        - Learning rates, batch sizes, and dataset mixtures differ by mode
        - Automatically resumes from the latest checkpoint if available
    """
    assert mode in ["focus", "flow", "finetune"], "Invalid mode"
    is_flow = mode == "flow"
    is_focus = mode == "focus"
    config = PRETRAIN_CONFIG if mode in ["focus", "flow"] else FINETUNE_CONFIG

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))
    model = RMKVModel(tokenizer.vocab_size_actual).to(device)

    # === Checkpoint resume logic ===
    start_step = 0
    ckpts = [f for f in os.listdir(PATHS["checkpoint_dir"]) if f.startswith(f"rmkv_{mode}_step") and f.endswith(".pt")]
    if ckpts:
        ckpts.sort(key=lambda f: int(re.findall(r'step(\\d+)', f)[0]))
        latest_ckpt = ckpts[-1]
        checkpoint_path = os.path.join(PATHS["checkpoint_dir"], latest_ckpt)
        if load_from_checkpoint(checkpoint_path, model, device):
            start_step = int(re.findall(r'step(\\d+)', latest_ckpt)[0])
            print(f"[Resume] Loaded checkpoint: {latest_ckpt} at step {start_step}")
        else:
            print(f"[Warning] Failed to load checkpoint: {latest_ckpt}. Starting from scratch.")

    # === Dataset and Dataloader setup ===
    if is_focus:
        dataset = InterleaveDataset(tokenizer, True, MODEL_CONFIG["max_seq_len"], weights=(5, 1, 1))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    elif is_flow:
        dataset = SegmentedDataset(
            tokenizer,
            source="fineweb",  # can later add interleaving here too
            segment_len=config["max_segment_len"],
            segments_per_sample=8
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    else:
        dataset = InterleaveDataset(tokenizer, False, MODEL_CONFIG["max_seq_len"], weights=(0, 2, 1))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"])

    # === Optimizer and Scheduler ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    if config.get("max_steps"):
        num_training_steps = config["max_steps"]
    else:
        approx_steps = 50000  # fallback
        num_training_steps = approx_steps * config["num_epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps
    )

    # === Start training ===
    train(
        model, dataloader, optimizer, scheduler,
        device, config, tokenizer.pad_token_id,
        tokenizer, mode, start_step=start_step
    )

    # === Save final model ===
    save_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{mode}_final.pt")
    save_checkpoint(model, optimizer, epoch=0, global_step=0, filepath=save_path)
    latest_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
    save_checkpoint(model, optimizer, epoch=0, global_step=0, filepath=latest_path)

    print(f"Model saved to {save_path} and {latest_path}")


if __name__ == "__main__":
    import argparse
    from training.checkpoint import load_from_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["focus", "flow", "finetune"], default="focus")
    args = parser.parse_args()
    main(args.mode)
