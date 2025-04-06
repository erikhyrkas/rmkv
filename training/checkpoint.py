# training/checkpoint.py
#
# Handles saving and loading model checkpoints for the RMKV model.
# Supports both model-only checkpoints and full training state preservation.
import os
import torch


def save_checkpoint(model, optimizer, epoch, global_step, filepath):
    """
    Save model and optimizer state to a checkpoint file.

    Creates a checkpoint containing the model weights, optimizer state,
    and training metadata to enable resuming training from this point.

    Args:
        model (nn.Module): The RMKV model instance to save
        optimizer (Optimizer): The optimizer used during training
        epoch (int): Current epoch number
        global_step (int): Current global training step count
        filepath (str): Path where the checkpoint will be saved

    Note:
        The saved checkpoint includes all information needed to resume
        training from exactly the same state, including learning rate
        scheduler position via the global_step.
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
    Load a model and optionally training state from a checkpoint.

    This function can be used in two modes:
    1. Simple model loading (when trainer=None)
    2. Full training state restoration (when trainer is provided)

    Args:
        checkpoint_path (str): Path to the checkpoint file
        model (nn.Module): Model to load the weights into
        device (torch.device, optional): Device to load the model onto
        trainer (object, optional): Trainer object to update with saved state
                                   (optimizer state, epoch count, etc.)

    Returns:
        bool: True if successfully loaded, False if failed to load

    Process:
        1. Load checkpoint file with appropriate device mapping
        2. Load model weights
        3. If trainer is provided, restore optimizer state, epoch/step counts
        4. For schedulers, fast-forward to the saved position

    Notes:
        - Handles both full checkpoints (dict with 'model_state_dict' key)
          and model-only checkpoints (just the state dict)
        - When loading for a different training mode (e.g., pretrain checkpoint
          for finetune), only the model weights are loaded
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # model.load_state_dict(checkpoint['model_state_dict'])
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

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