import os
import argparse
import torch
from config import PATHS, create_dirs
from model.rmkv import RMKVModel
from data.dataset import InstructionDataset
from training.trainer import Trainer
from training.checkpoint import load_from_checkpoint
import glob

def remove_files_by_pattern(directory, pattern):
    files_to_remove = glob.glob(os.path.join(directory, pattern))
    for file_path in files_to_remove:
        os.remove(file_path)
        print(f"Removed file: {file_path}")

def main():
    create_dirs()
    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["logs_dir"], exist_ok=True)

    parser = argparse.ArgumentParser(description="Train the RMKV model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = InstructionDataset()

    from data.tokenizer import RemarkableTokenizer
    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))

    model = RMKVModel(tokenizer.vocab_size_actual).to(device)
    trainer = Trainer(model, train_dataset, device)

    checkpoint_path = args.checkpoint or os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")

    if not args.checkpoint and os.path.exists(checkpoint_path):
        resume = input("Resume training? (Y/n) ").lower()
        if resume == "n":
            remove_files_by_pattern(PATHS["checkpoint_dir"], "rmkv_*.pt")

    if not load_from_checkpoint(checkpoint_path, model, device, trainer):
        print("Training from scratch")

    trainer.train()

if __name__ == "__main__":
    main()
