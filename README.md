![Remarkable Five](remarkable-five.png)
# Remarkable-Five

Remarkable-Five is an efficient, (yet to be determined) x.yB-parameter causal language model built using a novel RMKV (Recurrent Memory Key-Value) architecture. The project is designed to train stably on limited hardware (an Nvidia RTX 5000 with less than 30 GB of GPU memory), while achieving strong instruction-following performance specifically tailored for creative writing and story generation.

## Overview

Remarkable Five leverages a hybrid approach by combining Transformer-style QKV attention with a recurrent memory mechanism. This allows the model to:
- **Efficiently Handle Long Contexts:** Use a fixed number of memory tokens with learnable query vectors to capture long-range dependencies across text segments.
- **Support Extremely Long Sequences:** Optionally use sinusoidal positional encoding that can scale to sequences up to 128K tokens.
- **Implement Robust Causal Generation:** Use a sophisticated causal masking system where memory tokens can attend globally while regular tokens adhere to causal constraints.
- **Leverage Flexible Memory Updates:** Choose between GRU, LSTM, or a lightweight MinimalRNN cell for recurrent memory updates.
- **Train with Limited Resources:** Achieve competitive performance with a 1.2B-parameter model that can train on commodity hardware (an Nvidia RTX 5000 with less than 30 GB of GPU memory).
- **Excel at Creative Tasks:** The architecture is specifically optimized for instruction-following in creative writing and story generation tasks.

## Project Structure

```
remarkable_five/
├── README.md                   # This file -- Project overview and instructions
├── architecture.md             # Description of architecture
├── requirements.txt            # Python dependencies
├── config.py                   # Hyperparameters and configuration settings
├── complete.py                 # run the model to do completions
├── train.py                    # train the model
├── train_tokenizer.py          # train the tokenizer
├── checkpoints/                
│   ├── rmkv_pretrain_epoch_*.pt    # Pretraining checkpoints
│   ├── rmkv_finetune_epoch_*.pt    # Finetuning checkpoints
│   ├── rmkv_pretrain_latest.pt     # Latest pretraining checkpoint
│   ├── rmkv_finetune_latest.pt     # Latest finetuning checkpoint 
│   ├── rmkv_latest.pt              # General latest checkpoint (backward compatibility)
│   └── tokenizer.json              # result of tokenizer training
├── logs/                       
│   └── out.log                 # trainining logs (todo)
├── training_data/              
│   ├── *.txt                   # Training files
│   └── *.md                    # Training files
├── model/
│   ├── __init__.py
│   ├── rmkv.py                 # RMKV model architecture definition
│   ├── layers.py               # Custom RMKV layers (QKV, recurrent memory cell)
│   └── utils.py                # Helper functions for initialization and parameter counting
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Data loader for instruction-tuning data
│   └── tokenizer.py            # Custom ASCII-based tokenizer
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training loop and logging
│   ├── scheduler.py            # Learning rate scheduler helper
│   └── checkpoint.py           # Model checkpoint save/load functions
└── inference/
    ├── __init__.py
    └── infer.py                # Inference pipeline for text generation
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url> remarkable_five
   cd remarkable_five
   ```

2. **Install dependencies:**
   Use https://pytorch.org/get-started/locally/ to determine torch install, for me it is:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

3. **Create necessary directories:**
   Running the project (e.g., via `python config.py` or the training script) will create the needed directories for data, checkpoints, and logs.

## Usage

### Training
Train tokenizer:
```bash
python train_tokenizer.py
```

#### Pretraining
To start pretraining the RMKV model from scratch:
```bash
python train.py --mode pretrain
```

To resume pretraining from the latest checkpoint:
```bash
python train.py --mode pretrain
```
(You'll be prompted to confirm if you want to resume)

To restart pretraining from scratch (removing existing checkpoints):
```bash
python train.py --mode pretrain
```
(Answer 'Y' when prompted to restart)

#### Finetuning
To start finetuning after pretraining:
```bash
python train.py --mode finetune
```
This will automatically use the latest pretraining checkpoint.

To specify a particular checkpoint for finetuning:
```bash
python train.py --mode finetune --checkpoint checkpoints/rmkv_pretrain_epoch_10.pt
```

To continue finetuning from a previous finetuning run:
```bash
python train.py --mode finetune
```
(You'll be prompted to confirm if you want to resume)

To reset epoch counting when loading a checkpoint:
```bash
python train.py --mode finetune --checkpoint checkpoints/rmkv_pretrain_latest.pt --reset_epochs
```

#### Using Custom Data Directories
To specify a custom data directory:
```bash
python train.py --mode pretrain --data_dir /path/to/pretrain_data
python train.py --mode finetune --data_dir /path/to/finetune_data
```

### Inference
To generate text with a trained model:
```bash
python complete.py [--checkpoint <path_to_checkpoint>] [--prompt "hello world"] [--max_length=100]
```

## Training Configuration
The model uses different configurations for pretraining and finetuning:

- **Pretraining**: Higher learning rate, larger batch size, more epochs
- **Finetuning**: Lower learning rate, smaller batch size, fewer epochs, stricter gradient clipping

These configurations can be adjusted in `config.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas or improvements for the project.

## License

MIT

---