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
        global_step (int): Current global step.
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
    print(f"Checkpoint saved to {filepath}")


def load_from_checkpoint(checkpoint_path, model, device=None, trainer=None):
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the weights into
        device: Device to load the model onto
        trainer: Trainer object to update with saved state

    Returns:
        bool: True if successfully loaded, False otherwise
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])

            if trainer is not None:
                # Check if we're loading for the same mode or different mode
                is_same_mode = checkpoint_path.find(trainer.mode) != -1 or "rmkv_latest.pt" in checkpoint_path

                if trainer.optimizer is not None and is_same_mode:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    trainer.start_epoch = checkpoint['epoch'] + 1  # Start with next epoch
                    trainer.global_step = checkpoint.get('global_step', 0)

                    if trainer.scheduler is not None:
                        # Fast-forward the scheduler to the right position
                        for _ in range(trainer.global_step):
                            trainer.scheduler.step()

                    print(
                        f"Resuming {trainer.mode} training from epoch {trainer.start_epoch} (global step {trainer.global_step})")
                else:
                    print(
                        f"Loaded model weights only - starting new {trainer.mode} training from epoch {trainer.start_epoch}")

            return True

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False

    print(f"Checkpoint not found at {checkpoint_path}")
    return False