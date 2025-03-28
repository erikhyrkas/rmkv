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

# === Training Configuration ===
TRAINING_CONFIG = {
    "learning_rate": 1e-5,
    "batch_size": 8,                    # Adjust to fit within the RTX 5000 memory limits
    "weight_decay": 1e-2,
    "num_epochs": 20,
    "gradient_accumulation_steps": 4,   # Simulate larger batch sizes if needed
    "warmup_steps": 1000,               # Number of warmup steps for the learning rate scheduler
    "max_grad_norm": 1.0,               # For gradient clipping
    "fp16": True,                       # Enable mixed precision training
    "use_sincos": True,                 # use sin/cos based positional embedding
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

if __name__ == "__main__":
    create_dirs()
