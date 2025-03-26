# training/checkpoint.py

import torch


def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The RMKV model.
        optimizer (Optimizer): The optimizer used during training.
        epoch (int): Current epoch number.
        filepath (str): Path where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch} to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        model (nn.Module): The RMKV model.
        optimizer (Optimizer): The optimizer used during training.
        filepath (str): Path from where the checkpoint will be loaded.
        device (torch.device): Device to map the checkpoint.

    Returns:
        int: The epoch number from which to resume training.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {filepath}, resuming from epoch {epoch}")
    return epoch


if __name__ == "__main__":
    # Quick test: saving and loading a dummy checkpoint.
    import torch.nn as nn

    dummy_model = nn.Linear(10, 10)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    test_path = "dummy_checkpoint.pt"

    save_checkpoint(dummy_model, dummy_optimizer, epoch=1, filepath=test_path)
    load_epoch = load_checkpoint(dummy_model, dummy_optimizer, filepath=test_path, device=torch.device("cpu"))
    print("Resumed from epoch:", load_epoch)
