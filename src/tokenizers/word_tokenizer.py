"""
Script: word_tokenizer.py
Purpose: Train a Byte-Pair Encoding (Word) tokenizer on a text corpus, compute tokenization statistics, and save the tokenizer model, vocabulary and filtered dataset for downstream NLP tasks.
Inputs: Cleaned text dataset (CSV format)
Outputs: Trained Word tokenizer model (*.model), vocabulary file (*.vocab), tokenization statistics saved as JSON, histogram, and filtered dataset (*.csv)
Dependencies: sentencepiece, pandas, tqdm, numpy, matplotlib
"""

import json
import os
import re
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------


class WordTokenizer:
    """
    Wrapper around SentencePiece Word tokenizer.

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
        model_prefix: str = "Word_tokenizer",
        vocab_size: int = 32000,
    ):
        """
        Train a SentencePiece Word tokenizer on the provided corpus.

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
                model_type="word",
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


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------


def length_stats(series: pd.Series, name: str) -> dict:
    """
    Compute descriptive statistics for a numeric series (e.g., lengths of sequences).

    Args:
        series (pd.Series): Series of numeric values to compute statistics on.
        name (str): Label or name to associate with this series.

    Returns:
        dict: Dictionary containing:
            - 'name' (str): The provided name.
            - 'count' (int): Number of elements in the series.
            - 'mean' (float): Mean value.
            - 'std' (float): Standard deviation.
            - 'min' (int): Minimum value.
            - 'p50' (float): 50th percentile (median).
            - 'p80' (float): 80th percentile.
            - 'p90' (float): 90th percentile.
            - 'p95' (float): 95th percentile.
            - 'p99' (float): 99th percentile.
            - 'max' (int): Maximum value.
    """
    arr = series.to_numpy()

    stats = {
        "name": name,
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p80": float(np.percentile(arr, 80)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }

    return stats


def encode_stats(
    text: str, sp: spm.SentencePieceProcessor, unk: int
) -> tuple[int, int]:
    """
    Encode a string using a trained Word tokenizer and compute encoding statistics.

    Args:
        text (str): Input string to encode.
        sp (spm.SentencePieceProcessor): Loaded SentencePiece tokenizer.
        unk (int): ID of the unknown token.

    Returns:
        tuple[int, int]:
            - encoded_length (int): Number of token IDs generated.
            - unk_count (int): Number of unknown tokens (matching global `unk` ID).
    """
    ids = sp.Encode(text, out_type=int)
    return len(ids), ids.count(unk)


def split_text(text: str) -> str:
    tokens = re.findall(r"\d+|\w+|[^\w\s]", text, flags=re.UNICODE)
    return " ".join(tokens)


def main():
    # ------------------------------------------------------------
    # Loading the data
    # ------------------------------------------------------------

    # set up tqdm
    tqdm.pandas(ncols=100, dynamic_ncols=True)

    # set the root directory
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    processed_file_dir = os.path.join(ROOT_DIR, "data", "processed")

    # load the preprocessed data
    train_processed = pd.read_csv(
        os.path.join(processed_file_dir, "train_augmented.csv")
    )

    # split the text
    combined = (
        train_processed["cleaned_text"].astype(str)
        + " "
        + train_processed["highlights"].astype(str)
    )

    processed = combined.progress_apply(split_text)

    train_corpus = "\n".join(processed)

    # ------------------------------------------------------------
    # Setting up the Word tokenizer
    # ------------------------------------------------------------

    # Save to text corpus file for tokenizers
    model_prefix = "Word_tokenizer"
    vocab_size = 32000
    tokenizer = WordTokenizer()
    tokenizer.train(
        train_corpus,
        output_dir=os.path.join(ROOT_DIR, "tokenizer", "Word"),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
    )
    tokenizer.load(os.path.join(ROOT_DIR, "tokenizer", "Word", f"{model_prefix}.model"))

    unk = tokenizer.sp.unk_id
    # Create empty DataFrames with columns
    train_tokenized = pd.DataFrame()

    # ------------------------------------------------------------
    # Tokenizing and analyzing the tokenized data
    # ------------------------------------------------------------

    # Tokenize the articles
    train_tokenized[["token_article_len", "article_unk"]] = train_processed[
        "cleaned_text"
    ].progress_apply(lambda x: pd.Series(encode_stats(x, sp=tokenizer.sp, unk=unk)))
    # transfer the pre-tokenized text
    train_tokenized["cleaned_text"] = train_processed["cleaned_text"]
    train_tokenized["highlights"] = train_processed["highlights"]
    # Compute 90th percentile
    max_len = int(np.percentile(train_tokenized["token_article_len"], 90))

    # Optional: add 10% buffer
    threshold = int(max_len * 1.1)

    # Create mask (boolean Series) for articles <= max_len
    mask_train = train_tokenized["token_article_len"] <= threshold
    train_filtered = train_tokenized[mask_train].copy()

    # tokenize the highlights
    train_filtered[["token_highlight_len", "summary_unk"]] = train_processed[
        "highlights"
    ].progress_apply(lambda x: pd.Series(encode_stats(x, sp=tokenizer.sp, unk=unk)))

    # calculate stats of tokenized data
    article_stats = length_stats(train_filtered["token_article_len"], "articles")
    highlight_stats = length_stats(train_filtered["token_highlight_len"], "highlights")

    # make a histogram of tokenized lengths
    plt.hist(train_filtered["token_article_len"], bins=50)
    plt.title("Article Length Distribution")
    plt.savefig(os.path.join(ROOT_DIR, "data", "stats", "Word_article_length.png"))
    plt.close()

    # calculate OOV per sample
    oov_rate_articles: int = (
        train_filtered["article_unk"].sum() / train_filtered["token_article_len"].sum()
    )
    oov_rate_summary: int = (
        train_filtered["summary_unk"].sum()
        / train_filtered["token_highlight_len"].sum()
    )

    # calculate proper length of pre-tokenized sequences and calculate compression ratio
    WORD_RE = re.compile(r"\w+")
    train_filtered["cleaned_len"] = train_filtered["cleaned_text"].progress_apply(
        lambda x: len(WORD_RE.findall(x))
    )
    train_filtered["compression"] = (
        train_filtered["token_article_len"] / train_filtered["cleaned_len"]
    )

    # calculate compression data
    comp: pd.Series[float] = train_filtered["compression"]
    comp_mean_val = float(comp.mean())
    comp_std_val = float(comp.std())
    p50 = comp.quantile(0.5)
    p90 = comp.quantile(0.9)

    # make a compression distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(comp, bins=100, alpha=0.75)
    plt.axvline(
        comp_mean_val, linestyle="--", label=f"mean={comp_mean_val:.3f}", color="green"
    )
    plt.axvline(p50, linestyle=":", label=f"median={p50:.3f}", color="orange")
    plt.axvline(p90, linestyle="-.", label=f"p90={p90:.3f}", color="red")
    plt.legend()
    plt.xlabel("Compression Ratio")
    plt.ylabel("Density")
    plt.title("Word Compression Ratio Distribution")
    plt.savefig(
        os.path.join(ROOT_DIR, "data", "stats", "Word_compression_distribution.png")
    )
    plt.close()

    # ------------------------------------------------------------
    # Saving the results and filtered dataset
    # ------------------------------------------------------------

    # build a json data structure
    stats_path = os.path.join(ROOT_DIR, "data", "stats", "Word_stats.json")
    word_level_stats = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "vocab_size": int(tokenizer.sp.GetPieceSize()),
            "unk_id": int(tokenizer.sp.unk_id()),
        }
    }
    word_level_stats["length_filter"] = {
        "p90": int(max_len),
        "threshold": int(threshold),
        "kept_samples": int(len(train_filtered)),
        "total_samples": int(len(train_tokenized)),
    }
    word_level_stats["token_length_stats"] = {
        "article": article_stats,
        "highlight": highlight_stats,
    }
    word_level_stats["oov"] = {
        "articles_rate": float(oov_rate_articles),
        "summary_rate": float(oov_rate_summary),
        "total_article_unk": int(train_filtered["article_unk"].sum()),
        "total_summary_unk": int(train_filtered["summary_unk"].sum()),
    }
    word_level_stats["compression"] = {
        "mean": comp_mean_val,
        "std": comp_std_val,
        "median": p50,
        "p90": p90,
    }

    # save the json data
    with open(stats_path, "w") as f:
        json.dump(word_level_stats, f, indent=4)

    # save the tokenized data
    train_filtered[["cleaned_text", "highlights"]].to_csv(
        os.path.join(ROOT_DIR, "data", "processed", "Word_train.csv"), index=False
    )


if __name__ == "__main__":
    main()
