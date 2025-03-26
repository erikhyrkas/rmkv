# inference/infer.py

import argparse
import os

import torch
from model.rmkv import RMKVModel
from config import PATHS, MODEL_CONFIG
from training.checkpoint import load_checkpoint


def generate_text(model, tokenizer, prompt, device, max_length=100):
    """
    Autoregressively generate text given an input prompt.

    Args:
        model (nn.Module): The RMKVModel.
        tokenizer: Tokenizer instance with encode and decode methods.
        prompt (str): Input text prompt.
        device (torch.device): Computation device.
        max_length (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    model.eval()
    with torch.no_grad():
        # Encode the prompt using the tokenizer.
        input_ids = tokenizer.encode(prompt)
        # Convert to tensor and add batch dimension.
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = input_ids
        for _ in range(max_length):
            logits = model(generated)
            # Select the logits for the last generated token.
            next_token_logits = logits[:, -1, :]
            # Greedy decoding: choose the token with the highest logit.
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            # Assume token ID 0 is the end-of-sequence marker.
            if next_token.item() == 0:
                break

    # Decode the generated token IDs back into a string.
    output_text = tokenizer.decode(generated[0].tolist())
    return output_text


def main():
    parser = argparse.ArgumentParser(description="RMKV Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from data.tokenizer import RemarkableTokenizer
    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))
    actual_vocab_size = tokenizer.vocab_size_actual
    model = RMKVModel(actual_vocab_size, model_config=MODEL_CONFIG).to(device)

    # Create a dummy optimizer for loading the checkpoint (optimizer state is ignored during inference).
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0)
    load_checkpoint(model, dummy_optimizer, args.checkpoint, device)

    # Import the custom tokenizer (to be implemented in data/tokenizer.py).
    from data.tokenizer import RemarkableTokenizer  # Placeholder import
    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))

    output_text = generate_text(model, tokenizer, args.prompt, device, max_length=args.max_length)
    print("Generated Text:")
    print(output_text)


if __name__ == "__main__":
    main()
