# data/tokenizer.py
#
# Implements a custom BPE tokenizer for the RMKV model with special token support.
# This tokenizer handles consistent tokenization for both training and inference,
# with proper handling of whitespace, unicode normalization, and special tokens.
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD
from tokenizers.processors import TemplateProcessing
from config import PATHS
from tokenizers.pre_tokenizers import Sequence, ByteLevel, PreTokenizer
import re


class RemarkableTokenizer:
    """
    Custom tokenizer for the Remarkable Five model using BPE algorithm.

    This tokenizer combines byte-level BPE for robustness with special token
    handling for model control. It can be initialized with predefined settings
    or loaded from a saved tokenizer file.

    Args:
        vocab_size (int, optional): Size of vocabulary for training. Default: 30000
        special_tokens (list, optional): List of special tokens to include.
                                        Default: ["<think>", "</think>", "<start>",
                                                 "<end>", "<pad>", "<unk>"]
        load_path (str, optional): Path to a saved tokenizer file. If provided and
                                  the file exists, other arguments are ignored.

    Attributes:
        tokenizer: The underlying Hugging Face tokenizers implementation
        vocab_size: Target vocabulary size for training
        special_tokens: List of special tokens included in the vocabulary

    Note:
        The tokenizer handles whitespace consistently through ByteLevel pre-tokenizer
        and decoder, ensuring faithful text reconstruction after tokenization.
    """
    def __init__(self, vocab_size=30000, special_tokens=None, load_path=None):
        """
        Initialize the tokenizer either by loading from file or creating new.

        Args:
            vocab_size (int, optional): Target vocabulary size. Default: 30000
            special_tokens (list, optional): List of special tokens to include.
                                            Default: ["<think>", "</think>", "<start>",
                                                     "<end>", "<pad>", "<unk>"]
            load_path (str, optional): Path to a saved tokenizer file. If provided and
                                      exists, other arguments are ignored.

        Process:
            - If load_path is provided and exists, load the tokenizer from file
            - Otherwise, create a new BPE tokenizer with specified parameters
            - Configure normalizers, pre-tokenizers, and post-processors
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
            self.tokenizer.pre_tokenizer = Sequence([
                ByteLevel(add_prefix_space=False),
                Digits(individual_digits=True)
            ])

            # Use matching ByteLevel decoder to handle whitespace properly
            self.tokenizer.decoder = ByteLevelDecoder()

            self.tokenizer.post_processor = TemplateProcessing(
                single="$A",
                pair="$A $B",
                special_tokens=[(tok, idx) for idx, tok in enumerate(special_tokens)]
            )

    def train(self, files):
        """
        Train the tokenizer on a corpus of text files.

        Args:
            files (list or None): List of file paths to use for training.
                                 If None, uses all .txt and .md files from
                                 the configured data directory.

        Training process:
            1. Initializes a BPE trainer with the specified vocabulary size
            2. Ensures special tokens are included in the vocabulary
            3. Includes ByteLevel alphabet to handle all ASCII characters
            4. Trains on the provided files with progress display
        """
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
        """
        Convert text to token IDs.

        Args:
            text (str): Text to tokenize

        Returns:
            list: List of token IDs corresponding to the input text

        Note:
            This method handles all normalization and pre-tokenization steps
            configured for the tokenizer.
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids):
        """
        Convert token IDs back to text.

        Args:
            token_ids (list): List of token IDs to decode

        Returns:
            str: Reconstructed text from the token IDs

        Note:
            The ByteLevel decoder handles whitespace reconstruction correctly
            without needing manual replacements.
        """
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        """
        Save the tokenizer to disk.

        Args:
            path (str): File path where the tokenizer should be saved

        Note:
            The saved file includes all vocabulary, normalization rules,
            and special token configurations.
        """
        self.tokenizer.save(path)

    @property
    def pad_token_id(self):
        """
        Get the ID of the padding token.

        Returns:
            int: The ID of the padding token in the vocabulary
        """
        vocab = self.tokenizer.get_vocab()
        return vocab.get("<pad>")

    @property
    def end_of_response_id(self):
        """
        Get the ID of the end-of-response token.

        Returns:
            int: The ID of the <end> token in the vocabulary

        Note:
            This token marks the end of model-generated responses.
        """
        vocab = self.tokenizer.get_vocab()
        return vocab.get("<end>")

    @property
    def vocab_size_actual(self):
        """
        Get the actual size of the vocabulary after training.

        Returns:
            int: The number of tokens in the vocabulary

        Note:
            This may differ from the target vocab_size if the corpus
            was too small or if many special tokens were added.
        """
        return self.tokenizer.get_vocab_size()

    @property
    def vocabulary(self):
        """
        Get the full vocabulary as a token-to-ID mapping.

        Returns:
            dict: Dictionary mapping tokens to their IDs
        """
        return self.tokenizer.get_vocab()
