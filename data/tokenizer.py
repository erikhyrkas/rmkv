import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
from tokenizers.processors import TemplateProcessing
from config import PATHS


class RemarkableTokenizer:
    def __init__(self, vocab_size=30000, special_tokens=None, load_path=None):
        """
        If load_path is provided and exists, load the tokenizer from file.
        Otherwise, initialize a new tokenizer with the given parameters.
        """
        if load_path is not None and os.path.exists(load_path):
            self.tokenizer = Tokenizer.from_file(load_path)
        else:
            if special_tokens is None:
                special_tokens = ["<think>", "</think>", "<start>", "<end>", "<pad>", "<unk>"]
            self.vocab_size = vocab_size
            self.special_tokens = special_tokens

            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.tokenizer.decoder = BPEDecoder()
            self.tokenizer.post_processor = TemplateProcessing(
                single="$A",
                pair="$A $B",
                special_tokens=[(tok, idx) for idx, tok in enumerate(special_tokens)]
            )

    def train(self, files):
        if files is None:
            data_dir = PATHS["data_dir"]
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if f.endswith(".txt") or f.endswith(".md")]
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train(files, trainer)

    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def pad_token_id(self):
        vocab = self.tokenizer.get_vocab()
        return vocab.get("<pad>")

    @property
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    # Example usage for loading a trained tokenizer:
    tokenizer_save_path = os.path.join(PATHS["tokenizer_dir"], "tokenizer.json")

    # To load an existing tokenizer:
    tokenizer = RemarkableTokenizer(load_path=tokenizer_save_path)

    sample_text = "<think> What is a frog? </think> <start> An amphibian. <end>"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print("Original Text:", sample_text)
    print("Encoded Tokens:", encoded)
    print("Decoded Text:", decoded)
    print("Pad Token ID:", tokenizer.pad_token_id)
    print("Vocabulary Size:", tokenizer.vocab_size_actual)
