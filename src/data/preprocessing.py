"""
Script: preprocessing.py
Purpose: Clean text articles, compute statistics, generate histograms, and save cleaned datasets.
Inputs: raw train/validation/test CSV files
Outputs: cleaned CSVs and descriptive statistics/histograms in ../data/stats
Dependencies: pandas, tqdm, matplotlib, numpy
"""

import json
import os
import re
import string
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

tqdm.pandas(ncols=100, dynamic_ncols=True)


# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------


class CharTracker:
    """
    Tracks the number of characters removed during ASCII folding of text.

    Attributes:
        total_removed_chars (int): Cumulative count of characters removed
            during calls to `ascii_removal`.
    """

    def __init__(self):
        """
        Initialize a new CharTracker with zero removed characters.
        """
        self.total_removed_chars: int = 0

    def non_english_removal(self, text: str, allowed_chars: re.Pattern[str]) -> str:
        """
        Apply ASCII folding to a string and update the removal counter.

        This uses the `ascii_fold` function to remove non-ASCII characters,
        and keeps a running total of how many characters were removed.

        Args:
            text (str): The input text string to fold.
            allowed_chars (str): characters to leave in the string

        Returns:
            str: The ASCII-folded text.
        """
        folded: str = ascii_fold(text)
        cleared: str = remove_non_english(folded, allowed_chars=allowed_chars)
        self.total_removed_chars += len(text) - len(cleared)
        return cleared


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------


def remove_author(text: str) -> str:
    """
    Remove author byline patterns like 'By . Author Name .' at the start of the text.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with author byline removed.
    """
    return re.sub(r"^By\s+\.\s+.*?\.\s+", "", text)


def remove_cnn_markers(text: str) -> str:
    """
    Remove CNN-specific markers like '(CNN) --' and replace double dashes with single dash.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with CNN markers cleaned.
    """
    text: str = re.sub(r"\(CNN\)\s+--\s+", "", text)
    text = text.replace("--", "-")
    return text


def remove_published(text: str) -> str:
    """
    Remove lines starting with 'PUBLISHED:' and the following content until the newline.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with publication lines removed.
    """
    HEADER_RE: re.Pattern[str] = re.compile(
        r"""
    ^
    (?:By\s*\.\s*.*?\.\s*)?      # optional author block
    PUBLISHED:\s*\.\s*.*?\.\s*   # published metadata
    (?:\|\s*\.\s*)?              # optional separator
    UPDATED:\s*\.\s*.*?\.\s*     # updated metadata
    """,
        re.VERBOSE,
    )

    text: str = HEADER_RE.sub("", text).lstrip()
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in the text: collapse multiple spaces/tabs/newlines to single space.

    Args:
        text (str): The input text string.

    Returns:
        str: Text with normalized whitespace.
    """
    return " ".join(text.split())


def fix_punctuation(text: str) -> str:
    """
    Fix common punctuation spacing issues:
      - Remove space before punctuation (.,;!?)
      - Remove space after '(' and before ')'
      - Collapse multiple periods to a single period

    Args:
        text (str): The input text string.

    Returns:
        str: Text with cleaned punctuation.
    """
    text: str = re.sub(r"\s+([.,;!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text


def normalize_quotes(text: str) -> str:
    """
    Standardize quotes:
      - Convert curly quotes to straight quotes
      - Convert curly apostrophes to straight apostrophes

    Args:
        text (str): The input text string.

    Returns:
        str: Text with normalized quotes.
    """
    text: str = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return text


def remove_non_english(text: str, allowed_chars: re.Pattern[str]) -> str:
    """
    Remove special characters:
      - delete chinese characters, emojis, characters with accents and other special characters
      - only english characters, digits and punctuation are the remaining symbols

    Args:
        text (str): The input text string.
        allowed_chars (str): characters to leave in the string

    Returns:
        str: Text without special symbols
    """
    return allowed_chars.sub("", text)


def ascii_fold(text: str) -> str:
    """
    Apply ASCII folding to text to remove or substitute all non-ascii characters

    Args:
        text (str): The input text string.

    Returns:
        folded_text (str): ASCII-folded text
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def cleaning(
    text_input: str, tracker: CharTracker, allowed_chars: re.Pattern[str]
) -> str:
    """
    Full text cleaning pipeline that applies a sequence of preprocessing steps
    to standardize, normalize, and clean the input text.

    Steps:
      1. Remove author byline (e.g., 'By . Author Name .')
      2. Remove CNN-specific markers like '(CNN) --'
      3. Remove 'PUBLISHED:' lines and metadata
      4. Apply ASCII folding and track removed characters via `tracker`
      5. Remove any remaining non-English or special characters
      6. Normalize whitespace (collapse multiple spaces/newlines into single spaces)
      7. Fix common punctuation issues (spacing around punctuation, multiple periods)
      8. Normalize quotes (curly quotes → straight quotes, curly apostrophes → straight apostrophes)

    Args:
        text_input (str): The raw input text string to be cleaned.
        tracker (CharTracker): tracker of how many characters were removed
        allowed_chars (str): characters to leave in the string

    Returns:
        str: The fully cleaned and normalized text.
    """
    text: str = remove_author(text_input)
    text = remove_published(text)
    text = remove_cnn_markers(text)
    text = tracker.non_english_removal(text, allowed_chars=allowed_chars)
    text = normalize_whitespace(text)
    text = fix_punctuation(text)
    text = normalize_quotes(text)
    return text


def build_vocab(series: pd.Series) -> Counter:
    """
    Build a character-level vocabulary from a series of text.
      - Counts occurrences of each character in the series
      - Uses tqdm for progress visualization

    Args:
        series (pd.Series): Series of strings to process

    Returns:
        collections.Counter: Mapping from character to its frequency in the dataset
    """
    counter: Counter[str] = Counter()
    for text in tqdm(series, total=len(series)):
        counter.update(list(text))
    return counter


def cleaning_stats(raw_series: pd.Series, cleaned_series: pd.Series) -> dict:
    """
    Compute statistics on characters removed during cleaning.
      - Total raw characters
      - Total characters after cleaning
      - Number of characters removed
      - Percentage of characters removed

    Args:
        raw_series (pd.Series): Series of raw text
        cleaned_series (pd.Series): Series of cleaned text

    Returns:
        dict: Dictionary with keys:
            - 'total_raw_chars'
            - 'total_cleaned_chars'
            - 'chars_removed'
            - 'pct_removed'
    """
    stats: dict = {}
    raw_chars: int = raw_series.str.len().sum()
    cleaned_chars: int = cleaned_series.str.len().sum()
    stats["total_raw_chars"] = raw_chars
    stats["total_cleaned_chars"] = cleaned_chars
    stats["chars_removed"] = raw_chars - cleaned_chars
    stats["pct_removed"] = 100 * (raw_chars - cleaned_chars) / raw_chars
    return stats


def count_patterns(series: pd.Series, patterns: dict) -> dict:
    """
    Count occurrences of predefined patterns across a text series.
      - Uses a global `patterns` dictionary mapping pattern names to regexes
      - Returns total counts for each pattern in the series

    Args:
        series (pd.Series): Series of strings to analyze
        patterns (dict): Dict of strings of patterns to delete

    Returns:
        dict: Mapping of pattern name to total occurrence count
    """
    results: dict = {}
    for name, pat in patterns.items():
        results[name] = series.str.count(pat).sum()
    return results


def convert_np(obj):
    """
    Recursively convert NumPy data types to native Python types.

    This function traverses dictionaries and lists, converting any NumPy
    integers or floats to Python's built-in `int` or `float` types.
    Useful for preparing data to be serialized to JSON or other formats
    that don’t support NumPy types.

    Args:
        obj: Any object, potentially containing nested dicts/lists and NumPy scalars.

    Returns:
        The same structure as `obj`, but with NumPy integers/floats converted
        to native Python `int` and `float`.
    """
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj


def main():
    # ------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------

    # load the raw data
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    raw_path: str = os.path.join(ROOT_DIR, "data", "raw")
    processed_path: str = os.path.join(ROOT_DIR, "data", "processed")
    train: pd.DataFrame = pd.read_csv(os.path.join(raw_path, "train.csv"))
    val: pd.DataFrame = pd.read_csv(os.path.join(raw_path, "validation.csv"))
    test: pd.DataFrame = pd.read_csv(os.path.join(raw_path, "test.csv"))
    ALLOWED_CHARS: re.Pattern[str] = re.compile(
        rf"[^A-Za-z0-9\s{re.escape(string.punctuation)}]"
    )
    patterns: dict = {
        "double_hyphen": r"--",
        "cnn_marker": r"\(CNN\)",
        "published": r"PUBLISHED:",
        "curly_quotes": r"[“”‘’]",
    }
    run_stats: dict = {
        "raw_filtering": {},
        "cleaning": {},
        "patterns": {},
        "ascii_fold": {},
        "char_vocab": {},
    }

    # ------------------------------------------------------------
    # Sequence length analysis
    # ------------------------------------------------------------

    train["highlights"] = train["highlights"].str.replace("\n", " ", regex=False)
    val["highlights"] = val["highlights"].str.replace("\n", " ", regex=False)
    # analyze the length of the raw data
    train["char_len"] = train["article"].str.len()
    pre_clean_char_len: pd.Series[int] = train["char_len"]
    train["highlight_len"] = train["highlights"].str.len()
    train["highlight_ratio"] = train["highlight_len"] / train["char_len"]
    plt.hist(train["char_len"], bins=50)
    plt.title("Distribution of Article Character Lengths (Before Cleaning)")
    plt.savefig(
        os.path.join(ROOT_DIR, "data", "stats", "char_len_hist_before_cleaning.png")
    )
    plt.close()

    # save descriptive statistics for pre-cleaned data to CSV for later analysis
    pre_cleaning: pd.DataFrame = train[
        ["char_len", "highlight_len", "highlight_ratio"]
    ].describe()
    pre_cleaning.to_csv(
        os.path.join(ROOT_DIR, "data", "stats", "length_stats_pre_cleaning.csv"),
        index=True,
    )

    # clean the training data
    tracker: CharTracker = CharTracker()
    train["cleaned_text"] = train["article"].progress_apply(
        lambda x: cleaning(x, tracker, ALLOWED_CHARS)
    )

    # ------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------

    # filter out articles with character count <= 250 or highlight ratio outside [0.05, 0.5]
    train["char_len"] = train["cleaned_text"].str.len()
    train["highlight_len"] = train["highlights"].str.len()
    train["highlight_ratio"] = train["highlight_len"] / train["char_len"]

    # check the amount of samples violating each rule and the whole number of samples violating either of the rules
    mask_len: pd.Series[bool] = train["char_len"] > 250
    mask_ratio_min: pd.Series[bool] = train["highlight_ratio"] >= 0.05
    mask_ratio_max: pd.Series[bool] = train["highlight_ratio"] <= 0.5
    combined_mask: pd.Series[bool] = mask_len & mask_ratio_min & mask_ratio_max
    run_stats["raw_filtering"] = {
        "original_samples": int(len(train)),
        "removed_short": int((~mask_len).sum()),
        "removed_ratio_low": int((~mask_ratio_min).sum()),
        "removed_ratio_high": int((~mask_ratio_max).sum()),
        "remaining": int(combined_mask.sum()),
        "total_removed": int(len(train) - combined_mask.sum()),
    }

    # delete the samples
    mask: pd.Series[bool] = (
        (train["char_len"] > 250)
        & (train["highlight_ratio"] >= 0.05)
        & (train["highlight_ratio"] <= 0.5)
    )
    train = train[mask].copy()

    # plot the cleaned data
    plt.hist(train["char_len"], bins=50)
    plt.title("Distribution of Article Character Lengths (After Cleaning)")
    plt.savefig(
        os.path.join(ROOT_DIR, "data", "stats", "char_len_hist_after_cleaning.png")
    )
    plt.close()

    # save descriptive statistics for post-cleaned data to CSV for later analysis
    post_cleaning: pd.DataFrame = train[
        ["char_len", "highlight_len", "highlight_ratio"]
    ].describe()
    post_cleaning.to_csv(
        os.path.join(ROOT_DIR, "data", "stats", "length_stats_post_cleaning.csv"),
        index=True,
    )

    # define common bin edges for pre- / post-cleaning histograms to allow comparison
    bins: npt.NDArray[np.float64] = np.linspace(
        min(pre_clean_char_len.min(), train["char_len"].min()),
        max(pre_clean_char_len.max(), train["char_len"].max()),
        51,
    )

    # plot the overlay of both pre-cleaned and post-cleaned data
    plt.hist(
        pre_clean_char_len, bins=bins, color="blue", alpha=0.5, label="Pre-cleaning"
    )
    plt.hist(
        train["char_len"], bins=bins, color="red", alpha=0.5, label="Post-cleaning"
    )
    plt.title("Character Length Distribution: Pre vs Post Cleaning")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "data", "stats", "char_len_hist_comparison.png"))
    plt.close()

    # clean the validation data and delete outliers
    val["cleaned_text"] = val["article"].progress_apply(
        lambda x: cleaning(x, tracker, ALLOWED_CHARS)
    )
    val["char_len"] = val["cleaned_text"].str.split().progress_apply(len)
    val["highlight_len"] = val["highlights"].str.split().progress_apply(len)
    val["highlight_ratio"] = val["highlight_len"] / val["char_len"]
    mask = (
        (val["char_len"] > 250)
        & (val["highlight_ratio"] >= 0.05)
        & (val["highlight_ratio"] <= 0.5)
    )
    val = val[mask].copy()

    # save the processed data (without ids)
    train[["cleaned_text", "highlights"]].to_csv(
        os.path.join(processed_path, "train.csv"), index=False
    )
    val[["cleaned_text", "highlights"]].to_csv(
        os.path.join(processed_path, "validation.csv"), index=False
    )
    test[["article", "highlights"]].to_csv(
        os.path.join(processed_path, "test.csv"), index=False
    )

    # ------------------------------------------------------------
    # Character frequency analysis
    # ------------------------------------------------------------

    # build and print frequences of all characters present in the articles
    vocab_counter: Counter[str] = build_vocab(train["cleaned_text"])
    run_stats["char_vocab"] = {
        "unique_chars": len(vocab_counter),
        "top20": dict(
            sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }
    items: list[tuple[str, int]] = sorted(
        vocab_counter.items(), key=lambda x: x[1], reverse=True
    )
    chars: tuple[str]
    freqs: tuple[int]
    chars, freqs = zip(*items)
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(chars)), freqs)
    plt.xticks(range(len(chars)), chars)
    plt.xlabel("Character")
    plt.ylabel("Frequency")
    plt.title("Character Frequencies")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "data", "stats", "char_frequencies.png"))
    plt.close()

    # compute how many symbols were deleted during cleaning
    stats = cleaning_stats(train["article"], train["cleaned_text"])
    run_stats["cleaning"]["deletions"] = stats

    # compute how many patterns were deleted
    before = count_patterns(train["article"], patterns=patterns)
    after = count_patterns(train["cleaned_text"], patterns=patterns)
    impact = {k: before[k] - after[k] for k in before}
    run_stats["patterns"]["removed_counts"] = impact

    # track how many characters were normalized
    run_stats["ascii_fold"] = {
        "total_removed": int(tracker.total_removed_chars),
        "percent": float(
            100 * tracker.total_removed_chars / train["article"].str.len().sum()
        ),
    }
    run_stats["length_stats"] = {
        "pre_clean": pre_cleaning.to_dict(),
        "post_clean": post_cleaning.to_dict(),
    }

    # ------------------------------------------------------------
    # Data saving
    # ------------------------------------------------------------

    stats_path = os.path.join(ROOT_DIR, "data", "stats", "preprocess_stats.json")

    with open(stats_path, "w") as f:
        json.dump(convert_np(run_stats), f, indent=2)
    print("Saved run stats to:", stats_path)


if __name__ == "__main__":
    main()
