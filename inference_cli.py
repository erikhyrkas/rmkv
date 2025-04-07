# ./inference_cli.py
import torch
from model.rmkv import RMKVModel
from data.tokenizer import RemarkableTokenizer
from training.checkpoint import load_from_checkpoint
from config import PATHS
import os

# quick and dirty tool for testing.

MAX_TOKENS = 64

def main():
    tokenizer_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
    tokenizer = RemarkableTokenizer(load_path=tokenizer_path)

    model = RMKVModel(tokenizer.vocab_size_actual)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
    load_from_checkpoint(ckpt_path, model, device)

    model.eval()
    print("RMKV Inference Mode (Ctrl+C to exit)")

    while True:
        try:
            prompt = input("> ").strip()
            if not prompt:
                continue

            # Format like training: prompt + <start>, but only generate *after* that
            formatted_prompt = f"{prompt.strip()}<start>"
            input_ids = tokenizer.encode(formatted_prompt)
            generated = input_ids.copy()

            for _ in range(MAX_TOKENS):
                input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    logits_last = logits[0, -1]

                next_token = torch.argmax(logits_last).item()
                print(f"{next_token} -> {tokenizer.decode([next_token])}")

                if next_token == tokenizer.end_of_response_id:
                    break
                generated.append(next_token)

            output = tokenizer.decode(generated[len(input_ids):])
            print("\n=== COMPLETION ===")
            print(output.strip())
            print("==================\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
