import math
import torch
import torch.nn as nn

from config import MODEL_CONFIG


# -----------------------------------------------------------------------------
# Sin/Cos Positional Encoding (supports very long sequences, e.g. 128k tokens)
# -----------------------------------------------------------------------------
def get_sinusoidal_encoding(seq_len, dim):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape (1, seq_len, dim)


# -----------------------------------------------------------------------------
# Utility: Create a causal mask for the combined sequence (memory + tokens)
# -----------------------------------------------------------------------------
def get_causal_mask(memory_tokens, seq_len, device):
    """
    Returns a causal mask for a combined sequence of length (memory_tokens + seq_len).
    Memory tokens (indices [0, memory_tokens)) are unmasked.
    For token rows (indices [memory_tokens, memory_tokens+seq_len)),
    keys corresponding to future tokens (in the token part) are masked.
    The mask is of shape (memory_tokens+seq_len, memory_tokens+seq_len) and can be added
    to attention scores.
    """
    total_len = memory_tokens + seq_len
    # Start with an all-zero mask.
    mask = torch.zeros(total_len, total_len, device=device)

    # For rows corresponding to current tokens.
    # Create a lower triangular mask for the token part.
    # token_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # token_mask = token_mask * float('-inf')
    token_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * (-1e9)

    # For each token row, allowed keys:
    #   All memory tokens (first memory_tokens columns) are allowed.
    #   For token columns (the rest), allow only up to the current token.
    mask[memory_tokens:, memory_tokens:] = token_mask
    return mask.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, total_len, total_len)


# -----------------------------------------------------------------------------
# SigLU function (applies gating similar to GLU/SigLU)
# -----------------------------------------------------------------------------
def siglu(x):
    return x * torch.sigmoid(x)


# -----------------------------------------------------------------------------
# Minimal RNN Cell (as an alternative recurrent cell)
# -----------------------------------------------------------------------------
class MinimalRNNCell(nn.Module):
    """
    A minimal recurrent cell that uses a simple gating mechanism.
    Given previous state h and input summary s, it computes:
        new_h = h * gate + (1 - gate) * siglu(candidate)
    where gate and candidate are derived from a linear transformation
    of [h, s].
    """

    def __init__(self, embed_dim):
        super(MinimalRNNCell, self).__init__()
        self.linear = nn.Linear(2 * embed_dim, 2 * embed_dim)

    def forward(self, s, h):
        # s: input summary (batch, embed_dim)
        # h: previous memory (batch, embed_dim)
        combined = torch.cat([h, s], dim=-1)
        update = self.linear(combined)
        gate, candidate = update.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        candidate = siglu(candidate)
        new_h = h * gate + (1 - gate) * candidate
        return new_h

# -----------------------------------------------------------------------------
# Attention-based Pooling Module (Version 2)
# -----------------------------------------------------------------------------
class AttentionPooling2(nn.Module):
    """
    Uses a set of learnable query vectors (one per memory slot) to attend over the token outputs.
    Produces a weighted summary for each memory token.
    """

    def __init__(self, embed_dim, memory_tokens):
        super(AttentionPooling2, self).__init__()
        self.memory_tokens = memory_tokens
        # Each memory slot gets its own learnable query.
        self.query = nn.Parameter(torch.randn(memory_tokens, embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size = x.size(0)
        # Expand query to (batch, memory_tokens, embed_dim)
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        # Compute attention scores: (batch, memory_tokens, seq_len)
        scores = torch.bmm(query, x.transpose(1, 2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        # Compute weighted sum: (batch, memory_tokens, embed_dim)
        summary = torch.bmm(weights, x)
        return summary


# -----------------------------------------------------------------------------
# Recurrent Memory Cell 2
# -----------------------------------------------------------------------------
class RecurrentMemoryCell2(nn.Module):
    """
    Wraps several cell types (GRU, LSTM, or minimal) to update memory tokens.
    The interface always expects:
       h: previous memory state, s: input summary.
    """

    def __init__(self, embed_dim, cell_type="GRU"):
        super(RecurrentMemoryCell2, self).__init__()
        self.cell_type = cell_type.lower()
        if self.cell_type == "gru":
            self.cell = nn.GRUCell(embed_dim, embed_dim)
        elif self.cell_type == "lstm":
            self.cell = nn.LSTMCell(embed_dim, embed_dim)
        elif self.cell_type == "minimal":
            self.cell = MinimalRNNCell(embed_dim)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

    def forward(self, h, s):
        # h: (batch, embed_dim), s: (batch, embed_dim)
        if self.cell_type == "lstm":
            # For LSTM, initialize cell state to zeros.
            c0 = torch.zeros_like(h)
            new_h, _ = self.cell(s, (h, c0))
            return new_h
        else:
            return self.cell(s, h)


# -----------------------------------------------------------------------------
# QKV Block 2 with optional causal masking
# -----------------------------------------------------------------------------
class QKVBlock2(nn.Module):
    """
    A multi-head attention block that supports an optional attention mask.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(QKVBlock2, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, seq_len)

        # Apply causal mask if provided.
        if mask is not None:
            scores = scores + mask  # mask should broadcast to (batch, heads, seq_len, seq_len)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_linear(context)
        return out


# -----------------------------------------------------------------------------
# RMKV Block 2 with causal masking integrated
# -----------------------------------------------------------------------------
class RMKVBlock2(nn.Module):
    """
    Version 2 of the RMKV block.
    Now incorporates causal masking in the QKV block.
    """

    def __init__(self, embed_dim, num_heads, memory_tokens, dropout=0.1, cell_type="GRU"):
        super(RMKVBlock2, self).__init__()
        self.memory_tokens = memory_tokens
        self.qkv_block = QKVBlock2(embed_dim, num_heads, dropout)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_pool = AttentionPooling2(embed_dim, memory_tokens)
        self.recurrent_cell = RecurrentMemoryCell2(embed_dim, cell_type=cell_type)

    def forward(self, x, memory):
        """
        Args:
          x: (batch, seq_len, embed_dim) token representations.
          memory: (batch, memory_tokens, embed_dim) memory tokens.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Concatenate memory and tokens.
        combined = torch.cat([memory, x], dim=1)  # (batch, memory_tokens + seq_len, embed_dim)
        combined = self.ln(combined)

        # 2. Create a causal mask for the combined sequence.
        device = combined.device
        mask = get_causal_mask(self.memory_tokens, seq_len, device)  # shape: (1, 1, total, total)

        # 3. Process with QKV block using the mask.
        out = self.qkv_block(combined, mask=mask)
        out = self.dropout(out)

        # 4. Split output into memory and token parts.
        updated_memory = out[:, :self.memory_tokens, :]  # (batch, memory_tokens, embed_dim)
        new_x = out[:, self.memory_tokens:, :]  # (batch, seq_len, embed_dim)

        # 5. Use attention pooling over tokens to get summary per memory token.
        summary = self.attn_pool(new_x)  # (batch, memory_tokens, embed_dim)

        # 6. Update memory tokens with the recurrent cell.
        b, mt, d = updated_memory.shape
        updated_memory_flat = updated_memory.reshape(b * mt, d)
        summary_flat = summary.reshape(b * mt, d)
        updated_memory_flat = self.recurrent_cell(updated_memory_flat, summary_flat)
        updated_memory = updated_memory_flat.view(b, mt, d)

        # 7. Residual connection for token outputs.
        new_x = new_x + x

        return new_x, updated_memory


# -----------------------------------------------------------------------------
# RMKV Model 2
# -----------------------------------------------------------------------------
class RMKVModel2(nn.Module):
    """
    Version 2 of the RMKVModel.
    Supports:
      - Option to use sin/cos positional embeddings (e.g. for sequences up to 128k tokens)
      - Stacking RMKVBlock2 layers for long-range context
      - Configurable recurrent cell type in RMKVBlock2 -- minimal rnn option
    """

    def __init__(self, vocab_size, model_config=MODEL_CONFIG):
        super(RMKVModel2, self).__init__()
        self.model_config = model_config
        embed_dim = model_config["embed_dim"]

        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        max_seq_len = model_config["max_seq_len"]
        # Use sin/cos positional encoding if specified; otherwise, use learned embeddings.
        if model_config.get("use_sincos", False):
            self.pos_embed = get_sinusoidal_encoding(max_seq_len, embed_dim)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.memory_tokens = model_config["memory_tokens"]
        # Initialize memory tokens (learned and shared across the batch).
        self.initial_memory = nn.Parameter(torch.zeros(1, self.memory_tokens, embed_dim))
        nn.init.trunc_normal_(self.initial_memory, std=0.02)

        # Stack RMKVBlock2 layers.
        self.layers = nn.ModuleList([
            RMKVBlock2(
                embed_dim=embed_dim,
                num_heads=model_config["num_heads"],
                memory_tokens=self.memory_tokens,
                dropout=model_config["dropout"],
                cell_type=model_config["rnn_cell_type"]
            )
            for _ in range(model_config["num_layers"])
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids):
        """
        Args:
          input_ids: (batch, seq_len) tensor of token indices.
        Process:
          1. Compute token embeddings and add positional information.
          2. Initialize memory from learned initial_memory.
          3. Process through stacked RMKVBlock2 layers.
          4. Apply final normalization and project to logits.
        """
        batch_size, seq_len = input_ids.size()

        token_embeddings = self.token_embed(input_ids)  # (batch, seq_len, embed_dim)
        pos_embeddings = self.pos_embed[:, :seq_len, :]
        x = token_embeddings + pos_embeddings

        # Broadcast the initial memory for the batch.
        memory = self.initial_memory.expand(batch_size, -1, -1)

        # Process through each RMKVBlock2.
        for layer in self.layers:
            x, memory = layer(x, memory)

        x = self.ln_final(x)
        logits = self.head(x)
        return logits
