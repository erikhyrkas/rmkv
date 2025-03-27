# training/scheduler.py

import torch.optim.lr_scheduler as lr_scheduler
from config import SCHEDULER_CONFIG


def get_scheduler(optimizer, total_steps):
    """
    Create and return a learning rate scheduler for the given optimizer.

    Args:
        optimizer (Optimizer): The optimizer instance.
        total_steps (int): Total number of training steps for the scheduler.

    Returns:
        A learning rate scheduler instance.
    """
    scheduler_type = SCHEDULER_CONFIG.get("scheduler", "cosine").lower()

    if scheduler_type == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_type == "linear":
        # Define a simple linear decay lambda.
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: max(0.0, 1 - step / total_steps))
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


