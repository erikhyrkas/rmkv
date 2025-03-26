# Remarkable Five: The RMKV Architecture

**Remarkable Five** is an efficient, 1.2B-parameter causal language model built using a novel **RMKV (Recurrent Memory Key-Value)** architecture. This architecture combines transformer-style QKV attention with a recurrent memory module, offering a new balance of performance, efficiency, and long-term context handling.

RMKV stands for "Recurrent Memory Key-Value"—a fusion of attention-based short-term modeling and RNN-based long-term memory updating. By maintaining a persistent set of memory tokens that evolve across layers, RMKV captures contextual information with linear memory scaling.

---

## Architecture Overview

Each `RMKVBlock` contains:
1. A `QKVBlock` (Transformer-style attention over tokens + memory tokens)
2. A `RecurrentMemoryCell` (GRU or LSTM) for updating memory
3. LayerNorm and residual connections

### RMKVBlock Process:
- Concatenate memory and token embeddings
- Apply LayerNorm and QKV attention
- Split the result: updated memory and token features
- Mean-pool token features to summarize context
- Update memory tokens using a recurrent cell
- Apply residual connection to token stream

This design merges parallelizable training with efficient memory, making it suitable for long-context tasks and limited compute environments.

---

## Advantages of RMKV

- Long-context retention via fixed-size memory
- Efficient inference using compact recurrent updates
- Trains on commodity GPUs (e.g., RTX 5000 <30GB)
- Modular: you can plug in GRU or LSTM for memory updates
- Ideal for instruction tuning, story generation, and creative writing

The RMKV model balances architectural novelty with practical usability.

---

## Model Summary

- **Parameters:** ~1.2B
- **Layers:** Configurable (e.g., 12 or 24 `RMKVBlock`s)
- **Embedding Dim:** Defined in config
- **Memory Tokens:** Fixed-length, learned and recurrently updated
- **Positional Encoding:** Learnable
- **Tokenizer:** ASCII-based custom tokenizer

---

## Future Roadmap

- Synthetic data generation for instruction tuning
- Enhanced tokenization (subword / hybrid approaches)
- Fine-tuning with reinforcement learning (e.g., PPO)
- Inference optimizations: quantization, LoRA, distillation
- Expanded documentation and examples

