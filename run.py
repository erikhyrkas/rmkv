# ./run.py
#
# CLI tool for running inference with the RMKV model.
# Supports three modes of operation:
# 1. Chat mode: Maintains both memory state and conversation history between turns
# 2. Instruct mode: Maintains memory state but not conversation history
# 3. Single prompt mode: Processes one prompt and exits
import os
import argparse
import torch
from config import PATHS, MODEL_CONFIG
from model.rmkv import RMKVModel
from data.tokenizer import RemarkableTokenizer
from inference.infer import generate_text
from training.checkpoint import load_from_checkpoint


def main():
    """
    Entry point for RMKV inference CLI.

    Handles command-line argument parsing, model loading, and mode selection.
    Supports different inference modes:
    - chat: Interactive session with persistent memory and conversation history
    - instruct: Interactive session with persistent memory but separate prompts
    - single: Process a single prompt (specified via --prompt) and exit

    Command-line arguments:
        --checkpoint: Path to model checkpoint (defaults to latest)
        --prompt: Text prompt for single mode
        --max_length: Maximum tokens to generate
        --temperature: Sampling temperature (0 = deterministic)
        --top_p: Nucleus sampling threshold
        --mode: Inference mode (chat/instruct/single)
    """
    parser = argparse.ArgumentParser(description="Run inference with the RMKV model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max tokens to generate (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Sampling temperature; higher = more random (default: 0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability threshold (default: 0.9)")
    parser.add_argument("--mode", type=str, choices=["chat", "instruct", "single"], default="instruct",
                        help="Inference mode: chat (persistent memory & history), instruct (persistent memory only), or single (one-off prompt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = RemarkableTokenizer(load_path=tokenizer_path)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size_actual}")

    # Load model
    # Initialize with the vocabulary size from the tokenizer
    model = RMKVModel(tokenizer.vocab_size_actual).to(device)
    print(f"Initialized model with vocabulary size: {tokenizer.vocab_size_actual}")

    # Load checkpoint
    # First try the specified checkpoint, fall back to latest
    checkpoint_path = args.checkpoint or os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
    if not load_from_checkpoint(checkpoint_path, model, device):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Print model information
    # Report key metrics about the loaded model
    params = model.count_parameters()
    print(f"RMKV Model loaded successfully")
    print(f"Number of parameters: {params:,}")
    print(f"Context window: {MODEL_CONFIG['max_seq_len']} tokens")
    print(f"Memory tokens: {MODEL_CONFIG['memory_tokens']}")

    # Initialize memory (used in all modes)
    # This creates the initial memory state from the model's parameters
    memory = model.initial_memory.expand(1, -1, -1).to(device)

    # Handle different runtime modes
    if args.mode == "single" and args.prompt:
        # Single prompt mode (command line argument)
        # Process one prompt and exit - useful for scripts and automation
        run_single_prompt(model, tokenizer, device, args.prompt, memory, args.max_length, args.temperature, args.top_p)
    elif args.mode == "chat":
        # Chat mode - persistent history and memory
        # Best for conversational applications where context builds over time
        run_chat_mode(model, tokenizer, device, memory, args.max_length, args.temperature, args.top_p)
    else:
        # Default to instruct mode - persistent memory, non-persistent history
        # Good for instruction-following where each prompt is independent
        run_instruct_mode(model, tokenizer, device, memory, args.max_length, args.temperature, args.top_p)


def run_chat_mode(model, tokenizer, device, memory, max_length=4096, temperature=0.8, top_p=0.9):
    """
    Run the model in chat mode with persistent history and memory.

    In chat mode:
    - The entire conversation history is maintained between turns
    - The memory state evolves throughout the conversation
    - Each response builds on previous context

    This mode is ideal for chatbot applications where maintaining
    the conversation flow and context is important.

    Args:
        model: The loaded RMKV model
        tokenizer: Tokenizer for text encoding/decoding
        device: Computing device (CPU/GPU)
        memory: Initial memory state tensor
        max_length: Maximum number of tokens to generate per response
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold

    User Interface:
        - Enter text at the "You:" prompt
        - Type 'exit' to quit the chat session
        - Responses include the full conversation context
    """
    print("\nChat Mode: Type 'exit' to quit")
    print("Full conversation history and memory state are maintained between responses")

    # Initialize conversation history
    conversation = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        # Add <start> if not already present
        # This special token marks where the model should begin its response
        if "<start>" not in user_input:
            user_input = f"{user_input.strip()} <start>"

        conversation.append(user_input)

        # Prepare full conversation prompt
        # The model processes the entire conversation history each time,
        # allowing it to maintain coherent multi-turn interactions
        full_prompt = "\n".join(conversation)

        # Generate response with memory state
        # The memory persists between turns, accumulating information
        # while the return_memory=True ensures we get the updated memory back
        response, memory = generate_text(
            model, tokenizer, full_prompt, device,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            initial_memory=memory,
            return_memory=True,
            return_only_new_tokens=True
        )

        # Clean up response and display
        # Remove special tokens for cleaner output
        clean_response = response.replace("<start>", "").replace("<end>", "").strip()
        print(f"\nRMKV: {clean_response}")

        # Add response to history with <end> token for proper handling
        # Ensure the conversation history has proper special tokens
        if "<end>" not in response:
            response = f"{response} <end>"
        conversation.append(response)


def run_instruct_mode(model, tokenizer, device, memory, max_length=4096, temperature=0.8, top_p=0.9):
    """
    Run the model in instruct mode with persistent memory only.

    In instruct mode:
    - Each prompt is processed independently (no conversation history)
    - The memory state persists between prompts, maintaining long-term context
    - Each response is based on the current prompt and memory state only

    This mode is ideal for instruction-following where each prompt is
    a separate request, but benefits from accumulated context in memory.

    Args:
        model: The loaded RMKV model
        tokenizer: Tokenizer for text encoding/decoding
        device: Computing device (CPU/GPU)
        memory: Initial memory state tensor
        max_length: Maximum number of tokens to generate per response
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold

    User Interface:
        - Enter text at the ">" prompt
        - Type 'exit' to quit the instruct session
        - Each prompt is processed independently
    """

    print("\nInstruct Mode: Type 'exit' to quit")
    print("Memory is persistent between prompts, but conversation history is not")

    while True:
        prompt = input("\n> ")
        if prompt.lower() == "exit":
            break

        # Add <start> token if not present
        if "<start>" not in prompt:
            processed_prompt = f"{prompt.strip()} <start>"
        else:
            processed_prompt = prompt

        # Generate with persistent memory
        response, memory = generate_text(
            model, tokenizer, processed_prompt, device,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            initial_memory=memory,
            return_memory=True,
            return_only_new_tokens=True
        )

        # Display clean response
        clean_response = response.replace("<start>", "").replace("<end>", "").strip()
        print(f"\nRMKV: {clean_response}")


def run_single_prompt(model, tokenizer, device, prompt, memory, max_length=4096, temperature=0.8, top_p=0.9):
    """
    Run the model on a single prompt and exit.

    This mode processes one prompt non-interactively, typically used when:
    - Running from command line with the --prompt argument
    - Processing batch requests or integrating with other tools
    - Testing the model quickly with a specific input

    Args:
        model: The loaded RMKV model
        tokenizer: Tokenizer for text encoding/decoding
        device: Computing device (CPU/GPU)
        prompt: Text prompt to process
        memory: Initial memory state tensor
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold

    Output:
        Prints the cleaned model response to standard output
    """
    # Add <start> token if not already in the prompt
    # The <start> token signals to the model where to begin generating
    if "<start>" not in prompt:
        processed_prompt = f"{prompt.strip()} <start>"
    else:
        processed_prompt = prompt

    print(f"Generating up to {max_length} tokens...")

    # Generate with provided memory
    response, _ = generate_text(
        model, tokenizer, processed_prompt, device,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        initial_memory=memory,
        return_memory=True,
        return_only_new_tokens=True
    )

    # Clean and display response
    clean_response = response.replace("<start>", "").replace("<end>", "").strip()
    print(f"\nRMKV: {clean_response}")


if __name__ == "__main__":
    main()