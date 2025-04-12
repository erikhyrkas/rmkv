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


def load_for_inference(checkpoint_path, model, device=None):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return True

    print(f"Checkpoint not found at {checkpoint_path}")
    return False


def load_for_training(checkpoint_path, model, optimizer, scheduler=None, device=None, start_step=0):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading full training state from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            global_step = checkpoint.get('global_step', start_step)
            for _ in range(global_step - start_step):
                scheduler.step()

        return True, checkpoint.get('epoch', 0)

    print(f"Checkpoint not found at {checkpoint_path}")
    return False, 0
