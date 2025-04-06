# config.py

import os

# === Model Configuration for RMKV Architecture ===
MODEL_CONFIG = {
    "model_name": "RMKV",
    "num_layers": 12,         # Number of layers in the model
    "embed_dim": 2048,        # Token embedding dimension
    "num_heads": 8,           # Number of attention (or QKV-inspired) heads
    "dropout": 0.1,           # Dropout probability
    "max_seq_len": 1024,      # Maximum sequence length per segment
    "memory_tokens": 32,      # Number of memory tokens to carry forward between segments
    "rnn_cell_type": "GRU",   # Type of recurrent cell for memory updates (e.g., "GRU", "LSTM")
}

# === Pretraining Configuration ===
PRETRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "weight_decay": 1e-2,
    "num_epochs": 1,                    # used for train.py
    "max_steps": 5000,                  # used for train_hf.py -- 30000
    "gradient_accumulation_steps": 4,   # Simulate larger batch sizes
    "warmup_steps": 1000,               # Number of warmup steps for the learning rate scheduler
    "max_grad_norm": 1.0,               # For gradient clipping
    "fp16": True,                       # Enable mixed precision training
    "seed": 42,
}

# === Finetuning Configuration ===
FINETUNE_CONFIG = {
    "learning_rate": 5e-5,              # Lower learning rate for finetuning
    "batch_size": 4,                    # Smaller batch size for more precise updates
    "weight_decay": 1e-2,
    "num_epochs": 1,                    # Fewer epochs for finetuning
    "gradient_accumulation_steps": 4,
    "warmup_steps": 200,                # Fewer warmup steps for finetuning
    "max_grad_norm": 0.5,               # Stricter gradient clipping for stability
    "fp16": True,
    "seed": 42,
}

# Keep backward compatibility
TRAINING_CONFIG = PRETRAIN_CONFIG.copy()

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