# model/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class RecurrentMemoryCell(nn.Module):
    """
    A recurrent cell for updating memory tokens.
    Supports GRU or LSTM updates based on configuration.
    """
    def __init__(self, hidden_dim, cell_type="GRU"):
        super(RecurrentMemoryCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.upper()
        if self.cell_type == "GRU":
            self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTMCell(hidden_dim, hidden_dim)
        else:
            raise ValueError("Unsupported cell type: choose 'GRU' or 'LSTM'")

    def forward(self, memory, input_tensor):
        """
        Update memory using new input.
        Args:
            memory (Tensor or Tuple): Previous memory state(s) of shape (batch, hidden_dim).
                For LSTM, memory is a tuple (h, c).
            input_tensor (Tensor): New information to incorporate, shape (batch, hidden_dim).
        Returns:
            Updated memory state (Tensor or Tuple) with the same shape.
        """
        if self.cell_type == "GRU":
            updated_memory = self.cell(input_tensor, memory)
        elif self.cell_type == "LSTM":
            h, c = memory
            h_new, c_new = self.cell(input_tensor, (h, c))
            updated_memory = (h_new, c_new)
        return updated_memory

class QKVBlock(nn.Module):
    """
    A QKV-inspired block similar to Transformer attention.
    Projects the input to queries, keys, and values, applies scaled dot-product attention,
    and returns the attended output.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(QKVBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            mask (Tensor, optional): Attention mask of shape (batch_size, seq_length, seq_length)
        Returns:
            Tensor of shape (batch_size, seq_length, embed_dim) after attention.
        """
        batch_size, seq_length, embed_dim = x.size()

        # Linear projections to queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_length, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (batch, seq_length, num_heads, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, seq_length, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scale queries
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        # Compute attention scores: (batch, num_heads, seq_length, seq_length)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)
        # Reshape back: (batch, seq_length, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        return output
