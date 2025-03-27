# main.py

import argparse
import os
import torch
from config import PATHS, create_dirs
from model.rmkv import RMKVModel
import glob

def remove_files_by_pattern(directory, pattern):
    """
    Removes files in a directory that match a specified pattern.

    Args:
        directory (str): The path to the directory containing the files.
        pattern (str): The pattern to match (e.g., "*.txt", "file_*.log").
    """
    try:
        files_to_remove = glob.glob(os.path.join(directory, pattern))
        for file_path in files_to_remove:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    create_dirs()
    # Ensure necessary directories exist
    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["logs_dir"], exist_ok=True)

    parser = argparse.ArgumentParser(description="RMKV Project: Training, Inference, and Tokenizer Training")
    parser.add_argument("--mode", type=str, choices=["train", "infer", "train_tokenizer"], required=True,
                        help="Mode: 'train' for model training, 'infer' for inference, 'train_tokenizer' for training the tokenizer")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint (for resuming training or for inference)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Input prompt for inference mode")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum number of tokens to generate (in inference mode)")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size (default: 10000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        # Import the dataset loader from data/dataset.py
        from data.dataset import InstructionDataset
        train_dataset = InstructionDataset()  # Use default parameters or extend as needed

        model = RMKVModel(args.vocab_size).to(device)

        # Initialize trainer
        from training.trainer import Trainer
        trainer = Trainer(model, train_dataset, device)

        # load from a checkpoint to resume training.
        if args.checkpoint:
            target_checkpoint = args.checkpoint
        else:
            target_checkpoint = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
            if os.path.exists(target_checkpoint):
                resume = input("Resume training? (Y, n)")
                if resume == "n":
                    # remove all files named 'rmkv_*.pt' to clean
                    remove_files_by_pattern(PATHS["checkpoint_dir"], "rmkv_*.pt")

        from training.checkpoint import load_from_checkpoint
        if not load_from_checkpoint(target_checkpoint, model, device, trainer):
            print("Training from scratch")

        trainer.train()

    elif args.mode == "infer":
        model = RMKVModel(args.vocab_size).to(device)

        if not args.checkpoint:
            target_checkpoint = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
        else:
            target_checkpoint = args.checkpoint

        from training.checkpoint import load_from_checkpoint
        if not load_from_checkpoint(target_checkpoint, model, device):
            raise ValueError(f"No checkpoint found at {target_checkpoint}")

        from data.tokenizer import RemarkableTokenizer
        tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))

        # Simple greedy generation using the function from inference/infer.py
        from inference.infer import generate_text
        output_text = generate_text(model, tokenizer, args.prompt, device, args.max_length)
        print("Generated Text:")
        print(output_text)

    elif args.mode == "train_tokenizer":
        # Train the tokenizer using all files in PATHS['data_dir']
        from data.tokenizer import RemarkableTokenizer
        tokenizer = RemarkableTokenizer(vocab_size=args.vocab_size)
        # Calling train() with None will let the tokenizer automatically pick up all .txt files in PATHS["data_dir"]
        tokenizer.train(None)
        # Save the trained tokenizer model to a file (e.g., tokenizer.json in the data directory)
        tokenizer_save_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
        tokenizer.tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer trained and saved to {tokenizer_save_path}")


if __name__ == "__main__":
    main()
