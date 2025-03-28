import os
import argparse
import torch
from config import PATHS
from model.rmkv import RMKVModel
from data.tokenizer import RemarkableTokenizer
from inference.infer import generate_text
from training.checkpoint import load_from_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Run inference with the RMKV model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max_length", type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RemarkableTokenizer(load_path=os.path.join(PATHS["tokenizer_dir"], "tokenizer.json"))

    model = RMKVModel(tokenizer.vocab_size_actual).to(device)

    checkpoint_path = args.checkpoint or os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
    if not load_from_checkpoint(checkpoint_path, model, device):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    params = model.count_parameters()
    print(f"Number of parameters: {params:,}")

    if len(args.prompt) > 0:
        output_text = generate_text(model, tokenizer, args.prompt, device, args.max_length)

        print("Generated Text:")
        print(output_text)
    else:
        while True:
            prompt = input("> ")
            if prompt == "exit":
                break
            output_text = generate_text(model, tokenizer, prompt, device, args.max_length)
            print(output_text)


if __name__ == "__main__":
    main()
