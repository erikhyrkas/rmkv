# training/checkpoint.py
import os

import torch

from config import PATHS


def save_checkpoint(model, optimizer, epoch, global_step, filepath):
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The RMKV model.
        optimizer (Optimizer): The optimizer used during training.
        epoch (int): Current epoch number.
        filepath (str): Path where the checkpoint will be saved.
    """
    # Save checkpoint with our additional data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    torch.save(checkpoint, filepath)
    # Also save as latest
    latest_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
    torch.save(checkpoint, latest_path)
    print(f"Checkpoint saved to {filepath}")

def load_from_checkpoint(checkpoint_path, model, device=None, trainer=None):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if trainer is not None:
            if trainer.optimizer is not None:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.start_epoch = checkpoint['epoch'] + 1
                trainer.global_step = checkpoint.get('global_step', 0)

                if trainer.scheduler is not None:
                    # Fast-forward the scheduler to the right position
                    for _ in range(trainer.global_step):
                        trainer.scheduler.step()
            print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {trainer.start_epoch}")
        return True
    return False
