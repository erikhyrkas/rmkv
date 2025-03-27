import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD, StripAccents, Lowercase
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

            # Apply normalizers to handle unicode characters consistently
            self.tokenizer.normalizer = NFD()

            # Use ByteLevel pre-tokenizer with proper handling of whitespace
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

            # Use matching ByteLevel decoder to handle whitespace properly
            self.tokenizer.decoder = ByteLevelDecoder()

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
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=ByteLevel.alphabet()
        )
        self.tokenizer.train(files, trainer)

    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids):
        # ByteLevel decoder will handle whitespace correctly without manual replacement
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        """Save the tokenizer to disk"""
        self.tokenizer.save(path)

    @property
    def pad_token_id(self):
        vocab = self.tokenizer.get_vocab()
        return vocab.get("<pad>")

    @property
    def end_of_response_id(self):
        vocab = self.tokenizer.get_vocab()
        return vocab.get("<end>")

    @property
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocabulary(self):
        """Return the full vocabulary for inspection"""
        return self.tokenizer.get_vocab()