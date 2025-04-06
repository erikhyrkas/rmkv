import math
import torch
import torch.nn as nn

from config import MODEL_CONFIG

# -----------------------------------------------------------------------------
# RMKV Model: Recurrent Memory Key-Value Architecture
# -----------------------------------------------------------------------------
#
# This file implements a hybrid architecture combining:
# 1. Transformer-style QKV attention for parallelizable processing
# 2. Recurrent memory tokens that evolve across layers and contexts
# 3. Attention pooling to summarize sequence information for memory updates
#
# Key features:
# - Memory tokens can attend to all tokens (global attention)
# - Regular tokens use causal attention (can only see past tokens and memory)
# - Memory is updated recurrently using a choice of cell types (GRU/LSTM/Minimal)
# - Support for extremely long sequences through sinusoidal positional encodings
# - Efficient inference with fixed memory size regardless of context length
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Sin/Cos Positional Encoding (supports very long sequences, e.g. 128k tokens)
# -----------------------------------------------------------------------------
def get_sinusoidal_encoding(seq_len, dim):
    """
    Creates sinusoidal positional encodings for a sequence.

    Implements the positional encoding described in "Attention Is All You Need"
    with sine and cosine functions of different frequencies. This approach
    allows the model to generalize to sequence lengths not seen during training.

    Args:
        seq_len (int): Maximum sequence length
        dim (int): Dimension of the embedding space

    Returns:
        Tensor of shape (1, seq_len, dim) containing positional encodings

    Note:
        The output is unsqueezed to allow for easy broadcasting during
        addition to batched token embeddings.
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape (1, seq_len, dim)


# -----------------------------------------------------------------------------
# Utility: Create a causal mask for the combined sequence (memory + tokens)
# -----------------------------------------------------------------------------
def get_causal_mask(memory_tokens, seq_len, attention_mask=None, device=None):
    """
    Creates a combined causal attention mask for memory tokens and sequence tokens.

    This mask enables:
      1. Memory tokens to attend to all memory tokens (fully connected)
      2. Memory tokens to attend to all sequence tokens (cross-attention)
      3. Sequence tokens to attend to memory tokens (cross-attention)
      4. Sequence tokens to attend to previous sequence tokens only (causal self-attention)
      5. Proper masking of padding tokens when attention_mask is provided

    Args:
        memory_tokens (int): Number of memory tokens at the start of the sequence
        seq_len (int): Number of actual input tokens (excluding memory tokens)
        attention_mask: Optional tensor of shape (batch_size, seq_len) with properly
                       formatted values (0 for tokens, -1e9 for padding)
        device: Device for the mask tensor

    Returns:
        Tensor of shape (batch_size, 1, memory_tokens+seq_len, memory_tokens+seq_len)
        that can be added to attention scores. Values are 0 for allowed attention
        and -1e9 for positions that should not be attended to.
    """
    total_len = memory_tokens + seq_len

    if device is None and attention_mask is not None:
        device = attention_mask.device

    batch_size = 1
    if attention_mask is not None:
        batch_size = attention_mask.size(0)

    # Start with an all-zero mask
    mask = torch.zeros(batch_size, total_len, total_len, device=device)

    # Add causal masking only for the token part
    # This allows memory tokens to freely attend to each other
    token_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * (-1e9)
    mask[:, memory_tokens:, memory_tokens:] = token_mask

    # Incorporate attention_mask if provided
    # (assuming already formatted correctly with 0 for tokens, -1e9 for padding)
    if attention_mask is not None:
        # Create the padding mask for the combined sequence
        # Memory tokens are always valid
        combined_mask = torch.zeros(batch_size, total_len, device=device)
        # Set the token validity based on the attention_mask (already properly formatted)
        combined_mask[:, memory_tokens:] = attention_mask

        # Expand to full attention matrix with proper broadcasting
        padding_mask = combined_mask.unsqueeze(1) + combined_mask.unsqueeze(2)

        # Combine with the causal mask
        mask = mask + padding_mask

    return mask.unsqueeze(1)  # shape: (batch, 1, total_len, total_len)


# -----------------------------------------------------------------------------
# SigLU function (applies gating similar to GLU/SigLU)
# -----------------------------------------------------------------------------
def siglu(x):
    """
    Sigmoid-gated Linear Unit activation function.

    Computes x * sigmoid(x), similar to GLU but using the same tensor
    for both the value and the gate.

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Result of applying x * sigmoid(x)

    Note:
        This activation helps maintain gradient flow and introduces
        a form of gating with minimal parameter overhead.
    """
    return x * torch.sigmoid(x)


# -----------------------------------------------------------------------------
# Minimal RNN Cell (as an alternative recurrent cell)
# -----------------------------------------------------------------------------
class MinimalRNNCell(nn.Module):
    """
    A lightweight recurrent cell with simple gating mechanism.

    This cell provides an efficient alternative to GRU and LSTM for memory updates.
    Given previous state h and input summary s, it computes:
        new_h = h * gate + (1 - gate) * siglu(candidate)
    where gate and candidate are derived from a linear transformation of [h, s].

    Args:
        embed_dim (int): Dimension of the memory state

    Shape:
        - Input h: (batch_size, embed_dim) - previous memory state
        - Input s: (batch_size, embed_dim) - input summary
        - Output: (batch_size, embed_dim) - updated memory state

    Note:
        Uses a siglu activation (sigmoid-gated linear unit) for the candidate
        calculation, which helps maintain gradient flow during training.
    """

    def __init__(self, embed_dim):
        super(MinimalRNNCell, self).__init__()
        self.linear = nn.Linear(2 * embed_dim, 2 * embed_dim)

    def forward(self, h, s):
        # h: previous memory state (batch, embed_dim)
        # s: input summary (batch, embed_dim)
        combined = torch.cat([h, s], dim=-1)
        update = self.linear(combined)
        gate, candidate = update.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        candidate = siglu(candidate)
        new_h = h * gate + (1 - gate) * candidate
        return new_h


# -----------------------------------------------------------------------------
# Attention-based Pooling Module
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Attention-based pooling to create a contextual summary for memory updates.

    Uses a set of learnable query vectors (one per memory slot) to attend
    over the token representations, producing a weighted summary for each
    memory token separately.

    Args:
        embed_dim (int): Dimension of token and memory embeddings
        memory_tokens (int): Number of memory tokens (and thus query vectors)

    Shape:
        - Input x: (batch_size, seq_len, embed_dim) - token representations
        - Input attention_mask: Optional (batch_size, seq_len) - mask for padding
        - Output: (batch_size, memory_tokens, embed_dim) - context vectors for memory update

    Note:
        Each memory token gets its own learnable query vector, allowing specialized
        information gathering for different aspects of the input context.
    """

    def __init__(self, embed_dim, memory_tokens):
        super(AttentionPooling, self).__init__()
        self.memory_tokens = memory_tokens
        # Each memory slot gets its own learnable query.
        self.query = nn.Parameter(torch.randn(memory_tokens, embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, embed_dim)
        # attention_mask: (batch, seq_len) - optional mask for padding tokens
        batch_size = x.size(0)
        # Expand query to (batch, memory_tokens, embed_dim)
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        # Compute attention scores: (batch, memory_tokens, seq_len)
        scores = torch.bmm(query, x.transpose(1, 2)) * self.scale

        # Apply attention mask if provided (assume already in proper format)
        if attention_mask is not None:
            # Just add the mask directly - no conversion needed
            scores = scores + attention_mask.unsqueeze(1)  # (batch, 1, seq_len)

        weights = torch.softmax(scores, dim=-1)
        # Compute weighted sum: (batch, memory_tokens, embed_dim)
        summary = torch.bmm(weights, x)
        return summary


# -----------------------------------------------------------------------------
# Recurrent Memory Cell
# -----------------------------------------------------------------------------
class RecurrentMemoryCell(nn.Module):
    """
     Recurrent cell for updating memory tokens with contextual information.

     Wraps different RNN cell types with a consistent interface that always takes:
     - h: Previous memory state
     - s: Input summary (contextual information)

     Args:
         embed_dim (int): Dimension of memory embeddings
         cell_type (str): Type of cell to use: "gru", "lstm", or "minimal"

     Note:
         When using LSTM cells, only the hidden state (h) is tracked across
         iterations; the cell state (c) is reset to zeros for each update.
         This design choice prioritizes memory state consistency across cell types.
     """

    def __init__(self, embed_dim, cell_type="GRU"):
        super(RecurrentMemoryCell, self).__init__()
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
        """
        Forward pass through the recurrent cell.
        Args:
            h: previous memory state (batch, embed_dim)
            s: input summary (batch, embed_dim)
        Returns:
            new_h: updated memory state (batch, embed_dim)
        """
        if self.cell_type == "lstm":
            # For LSTM, initialize cell state to zeros.
            c0 = torch.zeros_like(h)
            new_h, _ = self.cell(s, (h, c0))
            return new_h
        elif self.cell_type == "gru":
            # GRU expects input first, then hidden state
            return self.cell(s, h)
        else:
            # Minimal RNN expects h then s
            return self.cell(h, s)


# -----------------------------------------------------------------------------
# QKV Block with optional causal masking
# -----------------------------------------------------------------------------
class QKVBlock(nn.Module):
    """
    Multi-head attention block for processing combined memory and token sequences.

    Implements standard transformer-style attention with query, key, and value
    projections, followed by scaled dot-product attention with optional masking.

    Args:
        embed_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability applied to attention weights

    Shape:
        - Input x: (batch_size, seq_len, embed_dim)
        - Input mask: Optional (batch_size, 1, seq_len, seq_len) containing
                     causal and padding masks combined
        - Output: (batch_size, seq_len, embed_dim)

    Note:
        The mask is additive and should contain 0 for positions to attend to
        and large negative values (e.g., -1e9) for positions to ignore.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(QKVBlock, self).__init__()
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
        # mask: (batch, 1, seq_len, seq_len) - includes both causal and padding masks
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, seq_len)

        # Apply causal and padding masks if provided.
        if mask is not None:
            scores = scores + mask  # mask should broadcast to (batch, heads, seq_len, seq_len)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_linear(context)
        return out


# -----------------------------------------------------------------------------
# RMKV Block with causal masking integrated
# -----------------------------------------------------------------------------
class RMKVBlock(nn.Module):
    """
    Core block of the RMKV architecture.

    Each RMKV block processes both token embeddings and memory tokens through:
    - Combined QKV attention with causal masking
    - Memory token updating via attention pooling and recurrent cells

    Args:
        embed_dim (int): Dimension of token and memory embeddings
        num_heads (int): Number of attention heads
        memory_tokens (int): Number of memory tokens
        dropout (float): Dropout probability
        cell_type (str): Type of recurrent cell ("GRU", "LSTM", or "minimal")

    Shape:
        - Input tokens: (batch_size, seq_len, embed_dim)
        - Input memory: (batch_size, memory_tokens, embed_dim)
        - Output tokens: (batch_size, seq_len, embed_dim)
        - Output memory: (batch_size, memory_tokens, embed_dim)
    """

    def __init__(self, embed_dim, num_heads, memory_tokens, dropout=0.1, cell_type="GRU"):
        super(RMKVBlock, self).__init__()
        self.memory_tokens = memory_tokens
        self.qkv_block = QKVBlock(embed_dim, num_heads, dropout)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_pool = AttentionPooling(embed_dim, memory_tokens)
        self.recurrent_cell = RecurrentMemoryCell(embed_dim, cell_type=cell_type)

    def forward(self, x, memory, attention_mask=None):
        """
           Process tokens and memory through one RMKV block.

           Args:
               x: Token representations of shape (batch_size, seq_len, embed_dim)
               memory: Memory tokens of shape (batch_size, memory_tokens, embed_dim)
               attention_mask: Optional mask of shape (batch_size, seq_len) with 1s for valid
                              tokens and 0s for padding.

           Returns:
               tuple: (new_x, updated_memory) where:
                   - new_x: Updated token representations (batch_size, seq_len, embed_dim)
                   - updated_memory: Updated memory tokens (batch_size, memory_tokens, embed_dim)

           Process flow:
               1. Concatenate memory and tokens for joint processing
               2. Apply LayerNorm for stability
               3. Create a causal mask for the combined sequence
               4. Process through QKV attention with the mask
               5. Split result back into memory and token components
               6. Use attention pooling to create summarized context for memory update
               7. Update memory tokens with the recurrent cell
               8. Apply residual connection for token outputs only
           """
        batch_size, seq_len, _ = x.shape

        # 1. Concatenate memory and tokens.
        combined = torch.cat([memory, x], dim=1)  # (batch, memory_tokens + seq_len, embed_dim)
        combined = self.ln(combined)

        # 2. Create a causal mask for the combined sequence, incorporating the attention mask
        device = combined.device
        mask = get_causal_mask(self.memory_tokens, seq_len, attention_mask, device)

        # 3. Process with QKV block using the mask.
        out = self.qkv_block(combined, mask=mask)
        out = self.dropout(out)

        # 4. Split output into memory and token parts.
        updated_memory = out[:, :self.memory_tokens, :]  # (batch, memory_tokens, embed_dim)
        new_x = out[:, self.memory_tokens:, :]  # (batch, seq_len, embed_dim)

        # 5. Use attention pooling over tokens to get summary per memory token.
        summary = self.attn_pool(new_x, attention_mask)  # (batch, memory_tokens, embed_dim)

        # 6. Update memory tokens with the recurrent cell.
        b, mt, d = updated_memory.shape
        updated_memory_flat = updated_memory.reshape(b * mt, d)
        summary_flat = summary.reshape(b * mt, d)

        # FIXED: Pass parameters in the correct order (h, s) per RecurrentMemoryCell's signature
        updated_memory_flat = self.recurrent_cell(updated_memory_flat, summary_flat)
        updated_memory = updated_memory_flat.view(b, mt, d)

        # 7. Residual connection for token outputs.
        new_x = new_x + x

        return new_x, updated_memory


# -----------------------------------------------------------------------------
# RMKV Model
# -----------------------------------------------------------------------------
class RMKVModel(nn.Module):
    """
    RMKV (Recurrent Memory Key-Value) language model.

    This model combines transformer-style attention with recurrent memory updates.
    Each layer processes both token embeddings and memory tokens, where:
    - Token embeddings follow a standard transformer flow with positional embeddings
    - Memory tokens are shared across all examples in a batch and evolve recurrently
    - Memory tokens can attend to all tokens, while regular tokens use causal attention

    Args:
        vocab_size (int): Size of vocabulary for token embeddings
        model_config (dict): Configuration dictionary with the following keys:
            - embed_dim (int): Dimension of token and memory embeddings
            - num_heads (int): Number of attention heads
            - num_layers (int): Number of RMKV blocks
            - max_seq_len (int): Maximum sequence length
            - memory_tokens (int): Number of memory tokens
            - dropout (float): Dropout probability
            - rnn_cell_type (str): Type of recurrent cell ("GRU", "LSTM", or "minimal")
            - use_sincos (bool): Whether to use sinusoidal positional encoding

    Shape:
        - Input: (batch_size, seq_len) token indices
        - Output: (batch_size, seq_len, vocab_size) logits
    """

    def __init__(self, vocab_size, model_config=MODEL_CONFIG):
        super(RMKVModel, self).__init__()
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

        # Add positional embedding for memory tokens to help with inter-memory attention
        self.memory_pos_embed = nn.Parameter(torch.zeros(1, self.memory_tokens, embed_dim))
        nn.init.trunc_normal_(self.memory_pos_embed, std=0.02)

        # Stack RMKVBlock layers.
        self.layers = nn.ModuleList([
            RMKVBlock(
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
        """
        Count the total number of trainable parameters in the model.

        Returns:
            int: Total number of parameters

        Note:
            This is useful for model size reporting and resource estimation.
        """
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the RMKV model.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token indices
            attention_mask: Optional tensor of shape (batch_size, seq_len) with properly
                           formatted values (0 for real tokens, -1e9 for padding).
                           Can be None during pretraining for efficiency.

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) containing output logits

        Process flow:
            1. Embed input tokens and add positional information
            2. Initialize memory from learned parameters (shared across batch)
            3. Process through stacked RMKV blocks, updating memory at each layer
            4. Apply final LayerNorm and project to vocabulary logits
        """
        # Ensure input_ids is a proper tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=next(self.parameters()).device)

        # Make sure it's 2D (batch, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension

        batch_size, seq_len = input_ids.size()

        token_embeddings = self.token_embed(input_ids)  # (batch, seq_len, embed_dim)
        pos_embeddings = self.pos_embed[:, :seq_len, :]

        if isinstance(pos_embeddings, torch.Tensor) and pos_embeddings.device != token_embeddings.device:
            pos_embeddings = pos_embeddings.to(token_embeddings.device)

        x = token_embeddings + pos_embeddings

        # Broadcast the initial memory for the batch and add positional embeddings
        memory = self.initial_memory.expand(batch_size, -1, -1)
        memory = memory + self.memory_pos_embed

        # Process through each RMKVBlock.
        for layer in self.layers:
            x, memory = layer(x, memory, attention_mask)

        x = self.ln_final(x)

        # If we have an attention mask, we should only compute logits for valid tokens
        logits = self.head(x)

        return logits

    def _ensure_tensor(self, token_ids, device=None):
        """
        Helper method to ensure token_ids is a properly formatted tensor.

        Handles conversion from various input formats (int, list, tensor) to a
        2D tensor suitable for model processing.

        Args:
            token_ids: Integer, list of integers, or tensor containing token IDs
            device: Target device for the tensor (defaults to model's device)

        Returns:
            2D tensor of shape (batch_size, seq_len) containing token IDs

        Note:
            Single integers and 1D sequences are automatically expanded to include
            a batch dimension.
        """
        if device is None:
            device = next(self.parameters()).device

        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor([token_ids] if isinstance(token_ids, int) else token_ids,
                                     dtype=torch.long, device=device)
        elif token_ids.device != device:
            token_ids = token_ids.to(device)

        # Make sure it's 2D (batch, seq_len)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # Add batch dimension

        return token_ids

    def generate_step(self, token_ids, memory):
        """
        Process a single generation step for autoregressive text generation.

        Takes the current token sequence and memory state, and returns next-token
        logits along with the updated memory state.

        Args:
            token_ids: List, tensor or single integer token ID(s)
                      Will be automatically converted to a proper tensor
            memory: Memory state tensor of shape (batch_size, memory_tokens, embed_dim)

        Returns:
            tuple: (logits, updated_memory) where:
                - logits: Tensor of shape (batch_size, seq_len, vocab_size)
                - updated_memory: Tensor of shape (batch_size, memory_tokens, embed_dim)

        Note:
            This method is optimized for inference and handles conversion of various
            input formats to the required tensor shape.
        """
        # Standardize token_ids to be a proper tensor
        token_tensor = self._ensure_tensor(token_ids)

        token_embeddings = self.token_embed(token_tensor)
        pos_embeddings = self.pos_embed[:, :token_tensor.size(1), :]

        if pos_embeddings.device != token_embeddings.device:
            pos_embeddings = pos_embeddings.to(token_embeddings.device)

        hidden_states = token_embeddings + pos_embeddings

        for layer in self.layers:
            hidden_states, memory = layer(hidden_states, memory, None)

        hidden_states = self.ln_final(hidden_states)
        logits = self.head(hidden_states)

        return logits, memory
