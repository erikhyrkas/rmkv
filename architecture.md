# Remarkable Five: The RMKV Architecture

**Remarkable Five** is an efficient, 1.2B-parameter causal language model built using a novel **RMKV (Recurrent Memory Key-Value)** architecture. This architecture combines transformer-style QKV attention with a recurrent memory module, offering a new balance of performance, efficiency, and long-term context handling.

RMKV stands for "Recurrent Memory Key-Value"—a fusion of attention-based short-term modeling and RNN-based long-term memory updating. By maintaining a persistent set of memory tokens that evolve across layers, RMKV captures contextual information with linear memory scaling.

---

## Architecture Overview

Each `RMKVBlock` contains:
1. A `QKVBlock` (Transformer-style multi-head attention with causal masking)
2. An `AttentionPooling` module (with learnable query vectors)
3. A `RecurrentMemoryCell` (GRU, LSTM, or MinimalRNN) for updating memory
4. LayerNorm and residual connections

### RMKVBlock Process:
1. Concatenate memory and token embeddings
2. Apply LayerNorm
3. Create a causal mask allowing:
   - Memory tokens to attend to all tokens
   - Regular tokens to attend only to memory and past tokens
4. Process through QKV attention using the causal mask
5. Split the result into updated memory and token features
6. Use attention pooling with learnable queries to summarize token context for each memory slot
7. Update memory tokens using a recurrent cell
8. Apply residual connection to token stream

This design merges parallelizable training with efficient memory, making it suitable for long-context tasks and limited compute environments.

---

## Key Improvements since inception

- **Enhanced Positional Encoding**: Option to use sinusoidal positional encodings that support very long sequences (up to 128K tokens)
- **Improved Causal Masking**: Dedicated masking system that properly enforces causality while allowing memory tokens to attend globally
- **Flexible RNN Cells**: Support for multiple recurrent cell types:
  - GRU (default)
  - LSTM
  - MinimalRNN (a lightweight custom recurrent cell with sigmoid gating)
- **Advanced Attention Pooling**: Each memory token now has its own learnable query vector for more targeted information capture
- **SigLU Activation**: Integration of SigLU (Sigmoid-weighted Linear Unit) for improved gating mechanisms

---

## Advantages of RMKV

- Long-context retention via fixed-size memory
- Efficient inference using compact recurrent updates
- Trains on commodity GPUs (e.g., RTX 5000 <30GB)
- Modular: you can plug in GRU, LSTM, or MinimalRNN for memory updates
- Support for extremely long contexts through sinusoidal positional encoding
- Ideal for instruction tuning, story generation, and creative writing

The RMKV model balances architectural novelty with practical usability.

---

## Model Summary

- **Parameters:** ~1.2B
- **Layers:** Configurable (e.g., 12 or 24 `RMKVBlock`s)
- **Embedding Dim:** Defined in config
- **Memory Tokens:** Fixed-length, learned and recurrently updated
- **Positional Encoding:** Option for learned or sinusoidal (for very long sequences)
- **Tokenizer:** ASCII-based custom tokenizer
- **RNN Cell Options:** GRU, LSTM, or MinimalRNN

---

## Technical Details

### Attention Mechanism
The `QKVBlock` implements multi-head attention with explicit scaling and dropout. It supports optional masking for causal attention, ensuring proper information flow.

### Memory Updating
Memory tokens are updated through a two-step process:
1. First pass through the attention mechanism alongside regular tokens
2. Then summarization of token information through `AttentionPooling`
3. Finally, recurrent update through the selected cell type

### Positional Information
The model can use either:
- Learned positional embeddings (standard approach)
- Sinusoidal positional encodings (for very long sequences)

### Parameter Efficiency
The model includes a `count_parameters` method for transparency about model size.

---

## Future Roadmap

- Synthetic data generation for instruction tuning
- Enhanced tokenization (subword / hybrid approaches)
- Fine-tuning with reinforcement learning (e.g., PPO)
- Inference optimizations: quantization, LoRA, distillation
- Expanded documentation and examples
- Exploration of other recurrent cell architectures
- Optimizations for very long context (>100K tokens)