"""
Script: tokenization.py
Purpose: Apply a called tokenizer to tokenize the data and save it.
Inputs: Cleaned text dataset (CSV format)
Outputs: Tokenized sequences in csv format
Dependencies: pandas, tqdm
"""

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from src.tokenizers.bpe_tokenizer import BPETokenizer


def save_token_count_json(token_count: int, filepath: str, overwrite: bool):
    """
    Save or append token count to a JSON file.

    Args:
        token_count (int): the value to save
        filepath (str): path to the JSON file
        overwrite (bool): if True, replace the file; if False, append to existing data
    """
    data = []

    if not overwrite and os.path.exists(filepath):
        # read existing data
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    # append new value
    data.append({"token_count": token_count})

    # write back
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def encode_with_special_tokens(text, tokenizer):
    ids = tokenizer.encode(text)
    return [tokenizer.sp.bos_id()] + ids + [tokenizer.sp.eos_id()]


def main(tokenizer_param: str, new_json: bool):
    tqdm.pandas(ncols=100, dynamic_ncols=True)

    # load the data
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    train_data = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", f"train_augmented.csv")
    )
    val_data = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", f"validation.csv")
    )
    test_data = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", f"test.csv"))

    # load the tokenizer
    TOKENIZER_REGISTRY = {
        "BPE": BPETokenizer,
    }
    tokenizer = TOKENIZER_REGISTRY[tokenizer_param]()
    tokenizer.load(
        os.path.join(
            ROOT_DIR,
            "tokenizer",
            f"{tokenizer_param}",
            f"{tokenizer_param}_tokenizer.model",
        )
    )

    # tokenize the train, validation and test sets
    train_data = train_data[["cleaned_text", "highlights"]].apply(
        lambda col: col.progress_apply(
            lambda x: encode_with_special_tokens(x, tokenizer)
        )
    )
    val_data = val_data[["cleaned_text", "highlights"]].apply(
        lambda col: col.progress_apply(
            lambda x: encode_with_special_tokens(x, tokenizer)
        )
    )
    test_data = test_data[["article", "highlights"]].apply(
        lambda col: col.progress_apply(
            lambda x: encode_with_special_tokens(x, tokenizer)
        )
    )

    # rename the columns
    train_data.columns = ["tokenized_articles", "tokenized_highlights"]
    val_data.columns = ["tokenized_articles", "tokenized_highlights"]
    test_data.columns = ["tokenized_articles", "tokenized_highlights"]

    # calculate total tokens in the set
    token_count: int = sum(
        len(x)
        for col in ["tokenized_articles", "tokenized_highlights"]
        for x in train_data[col]
    )
    json_path = os.path.join(ROOT_DIR, "data", "stats", "token_count.json")

    # suppose create_json is your boolean parameter
    save_token_count_json(token_count, json_path, overwrite=new_json)

    # save the data
    train_data.to_csv(
        os.path.join(
            ROOT_DIR, "data", "tokenized", f"tokenized_train_{tokenizer_param}.csv"
        ),
        index=False,
    )
    val_data.to_csv(
        os.path.join(
            ROOT_DIR, "data", "tokenized", f"tokenized_val_{tokenizer_param}.csv"
        ),
        index=False,
    )
    test_data.to_csv(
        os.path.join(
            ROOT_DIR, "data", "tokenized", f"tokenized_test_{tokenizer_param}.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BPE",
        choices=["BPE", "Word", "Canine"],
        help="Tokenizer regime to use",
    )
    parser.add_argument(
        "--new_json",
        action="store_true",
        help="Whether to create a new JSON for tokenized statistics (defaults to false)",
    )
    args = parser.parse_args()

    main(tokenizer_param=args.tokenizer, new_json=args.new_json)
