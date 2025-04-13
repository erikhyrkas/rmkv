# ./config.py

import os

TARGET_SEQUENCE_LENGTH = 4096

# === Model Configuration for RMKV Architecture ===
MODEL_CONFIG = {
    "model_name": "RMKV",
    "num_layers": 6,  # Number of layers in the model
    "embed_dim": 4096,  # Token embedding dimension
    "num_heads": 8,  # Number of attention (or QKV-inspired) heads
    "dropout": 0.1,  # Dropout probability
    "max_seq_len": 256,  # Maximum sequence length per segment
    "memory_tokens": 32,  # Number of memory tokens to carry forward between segments
    "rnn_cell_type": "minimal",  # Type of recurrent cell for memory updates (e.g., "GRU", "minimal")
}

# === Pretraining Configuration ===
FOCUS_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "weight_decay": 1e-2,
    "num_epochs": 1,  # used for train.py
    "max_steps": 50000,  # used for train_hf.py 8 x 256 x 50k = 102.4m tokens
    "gradient_accumulation_steps": 4,  # Simulate larger batch sizes
    "warmup_steps": 1000,  # Number of warmup steps for the learning rate scheduler
    "max_grad_norm": 1.0,  # For gradient clipping
    "fp16": True,  # Enable mixed precision training
    "max_segment_len": 256,  # Maximum sequence length per segment for phase 2 training
    "seed": 42,
}

FLOW_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 3,
    "weight_decay": 1e-2,
    "num_epochs": 1,  # used for train.py
    "max_steps": 16000000,  # used for train_hf.py 3 x 2048 x 16m = 6144 x 16m = 98.208b tokens
    "gradient_accumulation_steps": 4,  # Simulate larger batch sizes
    "warmup_steps": 1000,  # Number of warmup steps for the learning rate scheduler
    "max_grad_norm": 1.0,  # For gradient clipping
    "fp16": True,  # Enable mixed precision training
    "max_segment_len": 256,  # Maximum sequence length per segment for phase 2 training
    "seed": 42,
}

# === Finetuning Configuration ===
FINETUNE_CONFIG = {
    "learning_rate": 5e-5,  # Lower learning rate for fine-tuning
    "batch_size": 4,  # Smaller batch size for more precise updates
    "weight_decay": 1e-2,
    "max_steps": 1000000,  # 1m steps = about 16.3b tokens. possible steps: ~(22 million reasoning + 15 million nemotron) / 4 batch size = 9,250,000 steps to touch whole dataset
    "num_epochs": 1,  # Fewer epochs for fine-tuning
    "gradient_accumulation_steps": 4,
    "warmup_steps": 200,  # Fewer warmup steps for fine-tuning
    "max_grad_norm": 0.5,  # Stricter gradient clipping for stability
    "fp16": True,
    "max_segment_len": 256,  # max segments for fine-tuning
    "seed": 42,
}

# === Optimizer and Scheduler Settings ===
OPTIMIZER_CONFIG = {
    "optimizer": "AdamW",
}

SCHEDULER_CONFIG = {
    "scheduler": "cosine",  # Cosine learning rate decay
}

# === Directory Paths ===
PATHS = {
    "data_dir": os.path.join(os.getcwd(), "training_data"),
    "checkpoint_dir": os.path.join(os.getcwd(), "checkpoints"),
    "logs_dir": os.path.join(os.getcwd(), "logs"),
    "tokenizer_dir": os.path.join(os.getcwd(), "checkpoints"),
}


def create_dirs():
    """Create necessary directories if they do not exist."""
    print("Confirming directories...")
    for key, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
