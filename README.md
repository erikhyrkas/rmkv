# Remarkable Five

Remarkable Five is an efficient, 1.2B-parameter causal language model built using a novel RMKV (Recurrent Memory Key-Value) architecture. The project is designed to train stably on limited hardware (an Nvidia RTX 5000 with less than 30 GB of GPU memory), while achieving strong instruction-following performance specifically tailored for creative writing and story generation.

## Overview

Remarkable Five leverages a hybrid approach by combining ideas from Transformer QKV attention with a recurrent memory mechanism. This allows the model to:
- **Efficiently Handle Long Contexts:** Use a fixed number of memory tokens to capture long-range dependencies across text segments.
- **Support Causal Generation:** Autoregressively generate text using only past context.
- **Train with Limited Resources:** Achieve competitive performance on instruction-tuning tasks without the need for massive hardware or datasets.

## Project Structure

```
remarkable_five/
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
├── config.py                   # Hyperparameters and configuration settings
├── main.py                     # Entry point for training and inference
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
python main.py --mode train_tokenizer
```

To start training the RMKV model:
```bash
python main.py --mode train
```

### Inference
To generate text with a trained model:
```bash
python main.py --mode infer --checkpoint <path_to_checkpoint> --prompt "Your prompt here" --max_length 100
```

## Future Roadmap

- **Data Generation Module:** Implement advanced synthetic data generation pipelines.
- **Advanced Tokenization:** Experiment with subword tokenizers and novel tokenization strategies.
- **Fine-Tuning & RL:** Explore reinforcement learning and fine-tuning techniques for better instruction-following.
- **Optimization & Quantization:** Integrate inference optimizations such as quantization, LoRA, and distillation for deployment.
- **Extended Documentation:** Expand docs with detailed usage guides, API references, and design discussions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have ideas or improvements for the project.

## License

MIT

---

