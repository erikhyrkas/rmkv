# model/rmkv.py

import torch
import torch.nn as nn
from config import MODEL_CONFIG
from model.layers import QKVBlock, RecurrentMemoryCell


class RMKVBlock(nn.Module):
    """
    A single RMKV block that integrates a QKV-based transformer component with a recurrent memory update.
    It takes the input token representations and a set of memory tokens,
    concatenates them, applies a QKV block with layer normalization and dropout,
    and then updates the memory tokens using a recurrent memory cell.
    """

    def __init__(self, embed_dim, num_heads, memory_tokens, dropout=0.1, cell_type="GRU"):
        super(RMKVBlock, self).__init__()
        self.memory_tokens = memory_tokens
        self.qkv_block = QKVBlock(embed_dim, num_heads, dropout)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.recurrent_cell = RecurrentMemoryCell(embed_dim, cell_type=cell_type)

    def forward(self, x, memory):
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim) representing token embeddings.
            memory: Tensor of shape (batch, memory_tokens, embed_dim) representing current memory.
        Returns:
            new_x: Updated token representations (batch, seq_len, embed_dim).
            updated_memory: Updated memory tokens (batch, memory_tokens, embed_dim).
        Process:
          1. Concatenate memory and token embeddings.
          2. Normalize and process with the QKV block.
          3. Split the result into updated memory and new token outputs.
          4. Use a summary (mean-pooled token outputs) to update the memory via a recurrent cell.
          5. Apply a residual connection for the token outputs.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Concatenate memory and token embeddings along the sequence dimension.
        combined = torch.cat([memory, x], dim=1)  # (batch, memory_tokens + seq_len, embed_dim)
        combined = self.ln(combined)

        # 2. Process with the QKV block.
        out = self.qkv_block(combined)
        out = self.dropout(out)

        # 3. Split the output: first part for memory, remainder for tokens.
        updated_memory = out[:, :self.memory_tokens, :]  # (batch, memory_tokens, embed_dim)
        new_x = out[:, self.memory_tokens:, :]  # (batch, seq_len, embed_dim)

        # 4. Create a summary from new_x to update memory. Here we use mean pooling.
        summary = new_x.mean(dim=1)  # (batch, embed_dim)
        b, mt, d = updated_memory.shape
        # Expand the summary so each memory token gets the same update input.
        summary_expanded = summary.unsqueeze(1).expand(-1, mt, -1).reshape(b * mt, d)
        memory_flat = updated_memory.reshape(b * mt, d)
        updated_memory_flat = self.recurrent_cell(memory_flat, summary_expanded)
        updated_memory = updated_memory_flat.reshape(b, mt, d)

        # 5. Apply residual connection to token outputs.
        new_x = new_x + x

        return new_x, updated_memory

def get_sinusoidal_encoding(seq_len, dim):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape (1, seq_len, dim)

class RMKVModel(nn.Module):
    """
    The RMKVModel combines token embeddings, positional encodings, and a stack of RMKV blocks
    to produce a causal language model. It uses a learnable set of memory tokens that are updated
    across layers, enabling the model to capture longer context without a quadratic increase in cost.
    """

    def __init__(self, vocab_size, model_config=MODEL_CONFIG):
        super(RMKVModel, self).__init__()
        self.model_config = model_config
        embed_dim = model_config["embed_dim"]
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learned) for token positions up to max_seq_len.
        self.pos_embed = nn.Parameter(torch.zeros(1, model_config["max_seq_len"], embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # self.register_buffer("pos_embed", get_sinusoidal_encoding(model_config["max_seq_len"], embed_dim))

        self.memory_tokens = model_config["memory_tokens"]
        # Learned initial memory tokens (shared across the batch).
        self.initial_memory = nn.Parameter(torch.zeros(1, self.memory_tokens, embed_dim))
        nn.init.trunc_normal_(self.initial_memory, std=0.02)

        # Stack of RMKV blocks.
        self.layers = nn.ModuleList([
            RMKVBlock(embed_dim=embed_dim,
                      num_heads=model_config["num_heads"],
                      memory_tokens=self.memory_tokens,
                      dropout=model_config["dropout"],
                      cell_type=model_config["rnn_cell_type"])
            for _ in range(model_config["num_layers"])
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape (batch, seq_len) with token indices.
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size) representing output logits.
        Process:
          1. Embed input tokens and add positional embeddings.
          2. Initialize memory tokens for each batch from the learned initial memory.
          3. Process through each RMKV block, updating token representations and memory.
          4. Normalize final outputs and project to vocabulary space.
        """
        batch_size, seq_len = input_ids.size()

        # Embed tokens and add positional embeddings.
        token_embeddings = self.token_embed(input_ids)  # (batch, seq_len, embed_dim)
        pos_embeddings = self.pos_embed[:, :seq_len, :]
        # pos_embeddings = self.pos_embed[:, :seq_len, :].to(input_ids.device)
        x = token_embeddings + pos_embeddings

        # Initialize memory for each sample (broadcast the initial memory).
        memory = self.initial_memory.expand(batch_size, -1, -1)

        # Process through each RMKV block.
        for layer in self.layers:
            x, memory = layer(x, memory)

        x = self.ln_final(x)
        # with torch.no_grad():
        #     activation_norm = x.norm(dim=-1).mean().item()
        #     print(f"[Debug] Activation norm: {activation_norm:.4f}")

        logits = self.head(x)
        # with torch.no_grad():
        #     mem_norm = memory.norm(dim=-1).mean().item()
        #     print(f"[Debug] Memory Norm: {mem_norm:.4f}")
        return logits
