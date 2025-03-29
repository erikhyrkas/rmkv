import os
import argparse
import torch
from config import PATHS, create_dirs, PRETRAIN_CONFIG, FINETUNE_CONFIG
from data.dataset import InstructionDataset, EfficientPackedDataset
from model.rmkv import RMKVModel
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
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune"], default="pretrain",
                        help="Training mode: 'pretrain' for general language modeling or 'finetune' for instruction tuning")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory containing training data (defaults to config path)")
    parser.add_argument("--reset_epochs", action="store_true", help="Reset epoch counter when loading checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    from data.tokenizer import RemarkableTokenizer
    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))

    # Select dataset based on training mode
    data_dir = args.data_dir if args.data_dir else PATHS["data_dir"]
    if args.mode == "pretrain":
        print(f"Using EfficientPackedDataset for pretraining from {data_dir}")
        train_dataset = EfficientPackedDataset(data_dir=data_dir)
    else:  # finetune mode
        print(f"Using InstructionDataset for fine-tuning from {data_dir}")
        train_dataset = InstructionDataset(data_dir=data_dir)

    # Initialize model
    model = RMKVModel(tokenizer.vocab_size_actual).to(device)

    # Set up trainer with the appropriate mode
    trainer = Trainer(model, train_dataset, device, mode=args.mode)

    # Handle checkpoint loading
    checkpoint_provided = bool(args.checkpoint)
    config = PRETRAIN_CONFIG if args.mode == "pretrain" else FINETUNE_CONFIG

    # Determine which checkpoint to use
    if checkpoint_provided:
        checkpoint_path = args.checkpoint
    else:
        # Try to find a mode-specific checkpoint first
        mode_specific_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{args.mode}_latest.pt")
        if os.path.exists(mode_specific_path):
            checkpoint_path = mode_specific_path
            print(f"Using existing {args.mode} checkpoint: {checkpoint_path}")
        else:
            # Fall back to the latest checkpoint or pretrained model
            if args.mode == "finetune":
                # For finetuning, try to use the latest pretrain checkpoint if available
                pretrain_latest = os.path.join(PATHS["checkpoint_dir"], "rmkv_pretrain_latest.pt")
                if os.path.exists(pretrain_latest):
                    checkpoint_path = pretrain_latest
                    print(f"Using pretrained model for finetuning: {checkpoint_path}")
                else:
                    # If no pretrained model found, look for generic latest
                    checkpoint_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
            else:
                # For pretraining, use the generic latest
                checkpoint_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")

    # Ask whether to restart training if not providing a specific checkpoint
    if not checkpoint_provided and os.path.exists(checkpoint_path):
        num_epochs = config['num_epochs']
        final_epoch_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{args.mode}_epoch_{num_epochs}.pt")

        if os.path.exists(final_epoch_path):
            print(f"Found complete training run for {args.mode} mode (all {num_epochs} epochs).")
            resume = input(f"Restart {args.mode} training from scratch? (Y/n) ").lower()
            if resume != "n":
                # Remove all checkpoints for this mode
                remove_files_by_pattern(PATHS["checkpoint_dir"], f"rmkv_{args.mode}_*.pt")
                checkpoint_path = ""  # Set to empty to start from scratch
        else:
            resume = input(f"Resume {args.mode} training? (Y/n) ").lower()
            if resume == "n":
                # Remove all checkpoints for this mode
                remove_files_by_pattern(PATHS["checkpoint_dir"], f"rmkv_{args.mode}_*.pt")
                checkpoint_path = ""  # Set to empty to start from scratch

    # If we have a checkpoint path and it exists, try to load it
    if checkpoint_path and os.path.exists(checkpoint_path):
        if load_from_checkpoint(checkpoint_path, model, device, trainer):
            print(f"Successfully loaded checkpoint from {checkpoint_path}")

            # If reset_epochs is specified or we're switching from pretrain to finetune
            # (and not using a finetune-specific checkpoint), reset the epoch counter
            if (args.reset_epochs or
                    (args.mode == "finetune" and "pretrain" in checkpoint_path)):
                trainer.start_epoch = 0
                print(f"Reset epoch counter for {args.mode}")
    else:
        print(f"Training from scratch in {args.mode} mode")

    params = model.count_parameters()
    print(f"Number of parameters: {params:,}")

    trainer.train()


if __name__ == "__main__":
    main()