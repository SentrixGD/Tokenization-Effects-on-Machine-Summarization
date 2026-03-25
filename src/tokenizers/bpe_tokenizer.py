"""
Script: bpe_tokenizer.py
Purpose: Class of the bpe tokenizer
Dependencies: sentencepiece
"""

import os
import tempfile

import sentencepiece as spm


class BPETokenizer:
    """
    Wrapper around SentencePiece bpe tokenizer.

    Provides training, loading, encoding and decoding functionality.
    The tokenizer is uninitialized until `train()` or `load()` is called.
    """

    def __init__(self):
        """
        Initialize empty tokenizer state.
        """
        self.sp = None
        self.model_path = None

    def train(
        self,
        train_corpus: str,
        output_dir: str,
        model_prefix: str = "bpe_tokenizer",
        vocab_size: int = 32000,
    ):
        """
        Train a SentencePiece bpe tokenizer on the provided corpus.

        Args:
            train_corpus (str): Full training corpus as a single string.
            output_dir (str): Directory where model files will be saved.
            model_prefix (str): Prefix for generated model files.
            vocab_size (int): Vocabulary size.
        """
        os.makedirs(output_dir, exist_ok=True)
        prefix_path = os.path.join(output_dir, model_prefix)
        self.model_path = prefix_path + ".model"

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=True
        ) as tmp:
            tmp.write(train_corpus)
            tmp.flush()

            spm.SentencePieceTrainer.Train(
                input=tmp.name,
                model_prefix=prefix_path,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                shuffle_input_sentence=True,
                train_extremely_large_corpus=False,
                max_sentence_length=16384,
            )

        self.load(self.model_path)

    def load(self, model_path: str):
        """
        Load a pre-trained SentencePiece model.

        Args:
            model_path (str): Path to `.model` file.
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    def encode(self, text: str):
        """
        Encode text into token IDs.

        Args:
            text (str): Input text.

        Returns:
            list[int]: List of token IDs.
        """
        return self.sp.Encode(text)

    def decode(self, ids):
        """
        Decode token IDs back into text.

        Args:
            ids (list[int]): Token IDs.

        Returns:
            str: Decoded text.
        """
        return self.sp.Decode(ids)

    @property
    def unk_id(self):
        """
        Return the ID used for unknown tokens.

        Returns:
            int: Unknown token ID.
        """
        return self.sp.unk_id()
