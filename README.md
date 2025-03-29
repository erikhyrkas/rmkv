![Remarkable Five](remarkable-five.png)
# Remarkable Five

Remarkable Five is an efficient, (yet to be determined) x.yB-parameter causal language model built using a novel RMKV (Recurrent Memory Key-Value) architecture. The project is designed to train stably on limited hardware (an Nvidia RTX 5000 with less than 30 GB of GPU memory), while achieving strong instruction-following performance specifically tailored for creative writing and story generation.

## Overview

Remarkable Five leverages a hybrid approach by combining ideas from Transformer QKV attention with a recurrent memory mechanism. This allows the model to:
- **Efficiently Handle Long Contexts:** Use a fixed number of memory tokens to capture long-range dependencies across text segments.
- **Support Causal Generation:** Autoregressively generate text using only past context.
- **Train with Limited Resources:** Achieve competitive performance on instruction-tuning tasks without the need for massive hardware or datasets.

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
│   ├── rmkv_epoch_*.py         # previous model checkpoints
│   ├── rmkv_latest.py          # last model checkpoint
│   └── tokenizer.json          # result of tokenizer training
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

To start training the RMKV model:
```bash
python train.py
```

### Inference
To generate text with a trained model:
```bash
python complete.py [--checkpoint <path_to_checkpoint>] [--prompt "hello world"] [--max_length=100]
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas or improvements for the project.

## License

MIT

---

