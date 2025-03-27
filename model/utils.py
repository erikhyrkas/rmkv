# model/utils.py

import torch.nn as nn


def init_weights(module):
    """
    Initialize weights of the module.
    For linear layers, uses Xavier uniform initialization.
    For embeddings, uses a normal distribution with a small standard deviation.
    For LayerNorm, initializes weights to 1 and biases to 0.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
