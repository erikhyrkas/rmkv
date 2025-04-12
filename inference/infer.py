# inference/infer.py
#
# This module implements autoregressive text generation for the RMKV model.
# It handles sampling, temperature control, top-p filtering, repetition penalties,
# and other generation controls to produce high-quality text completions.
import torch
import torch.nn.functional as F


def generate_text(model, tokenizer, prompt, device, max_length=4096, temperature=0.8, top_p=0.9,
                  initial_memory=None, return_memory=False, return_only_new_tokens=False,
                  repetition_penalty=1.2, repetition_window=5):
    """
    Autoregressively generate text given an input prompt with long-context support.

    This function handles the complete text generation process, including:
    - Processing initial prompt through the model
    - Autoregressive token generation with sampling
    - Repetition avoidance and dynamic temperature adjustment
    - Special token handling (e.g., end tokens)

    Args:
        model (nn.Module): Trained RMKV model instance
        tokenizer: Tokenizer with encode/decode methods and vocabulary access
        prompt (str): Input text prompt to continue from
        device (torch.device): Computation device (CPU or GPU)
        max_length (int): Maximum number of new tokens to generate
        temperature (float): Sampling temperature; higher values increase randomness
                            Set to 0 for greedy (deterministic) decoding
        top_p (float): Nucleus sampling threshold (0.0-1.0); lower values increase focus
        initial_memory (Tensor, optional): Initial memory state. If None, uses model's
                                          default initial memory
        return_memory (bool): If True, returns the final memory state along with text
        return_only_new_tokens (bool): If True, returns only newly generated tokens
                                      If False, returns prompt + generated tokens
        repetition_penalty (float): Penalty applied to recently generated tokens
                                   Values > 1.0 reduce repetition
        repetition_window (int): Number of recent tokens to consider for repetition penalty

    Returns:
        str or tuple: If return_memory is False, returns the generated text
                     If return_memory is True, returns (generated_text, final_memory)

    Example:
        >>> model = RMKVModel(vocab_size).to(device)
        >>> tokenizer = RemarkableTokenizer(load_path="tokenizer.json")
        >>> generated_text = generate_text(
        ...     model, tokenizer, "Once upon a time", device,
        ...     max_length=100, temperature=0.7
        ... )
    """
    model.eval()
    max_seq_len = model.model_config["max_seq_len"]

    # Initialize special token handling
    # Get special token IDs for controlling generation
    special_token_ids = set()
    if hasattr(tokenizer, 'vocabulary'):
        for token in ['<start>', '<end>', '<think>', '</think>', '<pad>', '<unk>']:
            if token in tokenizer.vocabulary:
                special_token_ids.add(tokenizer.vocabulary[token])

    # Pre-compute end token ID for faster checking during generation
    end_token_id = None
    if hasattr(tokenizer, 'end_of_response_id'):
        end_token_id = tokenizer.end_of_response_id
    elif hasattr(tokenizer, 'vocabulary') and "<end>" in tokenizer.vocabulary:
        end_token_id = tokenizer.encode("<end>")[0]

    with torch.no_grad():
        # Encode the prompt using the tokenizer
        input_ids = tokenizer.encode(prompt)

        # Keep track of all generated token IDs
        all_token_ids = input_ids.copy()

        memory = initial_memory

        # Process input prompt in segments if needed (for prompts > max_seq_len)
        # This allows handling arbitrarily long prompts by processing them in chunks
        # while maintaining the memory state across chunks
        for start_idx in range(0, len(input_ids), max_seq_len):
            end_idx = min(start_idx + max_seq_len, len(input_ids))
            segment = input_ids[start_idx:end_idx]

            # Process segment using the model's generate_step method
            with torch.inference_mode():
                # We only need the updated memory from processing the prompt
                _, memory = model.generate_step(segment, memory)

        # Track newly generated tokens separately if needed
        generated_ids = []

        # Track repetition
        repetition_count = 0
        last_tokens = []

        # Begin autoregressive generation
        # The generation algorithm uses dynamic temperature adjustment and
        # repetition penalties to improve output quality
        for i in range(max_length):
            # Take the last max_seq_len-1 tokens to make room for the next generation
            recent_tokens = all_token_ids[-max_seq_len + 1:] if len(all_token_ids) >= max_seq_len else all_token_ids

            with torch.inference_mode():
                # Get logits and updated memory from model
                logits, memory = model.generate_step(recent_tokens, memory)

                # Get logits for the last token
                next_token_logits = logits[0, -1, :].clone()  # Clone to avoid modifying original tensor

                # Apply repetition penalty to recent tokens
                # This reduces the probability of generating the same tokens again
                for token_id in set(recent_tokens[-repetition_window:]):
                    if token_id in recent_tokens[-3:]:  # Extra penalty for very recent repeats
                        next_token_logits[token_id] /= repetition_penalty * 1.5
                    else:
                        next_token_logits[token_id] /= repetition_penalty

                # Apply a stronger penalty to special tokens (except end token)
                for token_id in special_token_ids:
                    if token_id != end_token_id:
                        next_token_logits[token_id] /= 1.5

                # Enhance end token probability after a reasonable length
                if i > 10 and end_token_id is not None:
                    next_token_logits[end_token_id] *= 1.05

                # Dynamic temperature adjustment
                current_temp = temperature
                # If we've generated the same token multiple times in a row, increase temperature
                # to encourage diversity in the output
                if len(last_tokens) >= 3 and len(set(last_tokens[-3:])) == 1:
                    current_temp = min(temperature * 1.5, 1.5)  # Increase but cap at 1.5
                    repetition_count += 1
                else:
                    repetition_count = 0

                # Emergency break for extreme repetition
                # This prevents the model from getting stuck in a repetition loop
                if repetition_count > 10:
                    if end_token_id is not None:
                        all_token_ids.append(end_token_id)
                        generated_ids.append(end_token_id)
                    break

                # Greedy decoding (argmax) if temperature is 0
                if current_temp == 0:
                    next_token = torch.argmax(next_token_logits).item()
                else:
                    # Apply temperature
                    next_token_logits = next_token_logits / current_temp

                    # Apply top-p (nucleus) sampling
                    # This restricts sampling to the smallest set of tokens whose cumulative
                    # probability exceeds top_p, preventing sampling from low-probability tokens
                    if top_p < 1.0:
                        # Sort logits in descending order
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        # Calculate cumulative probabilities
                        sorted_probs = F.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep the first token above threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        # Set removed indices to large negative value
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('Inf')

                    # Get probabilities and sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

            # Update tracking for repetition detection
            last_tokens.append(next_token)
            if len(last_tokens) > repetition_window:
                last_tokens.pop(0)

            # Add the next token to tracking collections
            all_token_ids.append(next_token)
            generated_ids.append(next_token)

            # Check for end conditions
            if end_token_id is not None and next_token == end_token_id:
                break

        # Determine which tokens to decode based on return_only_new_tokens
        tokens_to_decode = generated_ids if return_only_new_tokens else all_token_ids
        output_text = tokenizer.decode(tokens_to_decode)

        if return_memory:
            return output_text, memory
        else:
            return output_text