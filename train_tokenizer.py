import os
import argparse
from config import PATHS
from data.tokenizer import RemarkableTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a new tokenizer")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    args = parser.parse_args()

    tokenizer = RemarkableTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(None)

    tokenizer_save_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")
    tokenizer.tokenizer.save(tokenizer_save_path)
    print(f"Tokenizer trained and saved to {tokenizer_save_path}")

if __name__ == "__main__":
    main()
