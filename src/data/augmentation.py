"""
Script: augmentation.py
Purpose: Augment and expand the dataset.
Inputs: processed filtered training set
Outputs: augmented training set
Dependencies: spacy, pandas, tqdm, nltk, matplotlib, numpy
"""

import argparse
import ast
import os
import random
from collections import Counter
from functools import lru_cache
from typing import Any, Mapping, Sequence

import nltk
import numpy.typing as npt
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
from spacy.language import Language
from spacy.tokens import Doc, Token
from tqdm import tqdm

tqdm.pandas(ncols=100, dynamic_ncols=True)
random.seed(42)


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------
@lru_cache(maxsize=5000)
def cached_synonyms(lemma: str, wn_pos: str) -> tuple[str]:
    """
    Generates a list of synonyms for a given word based on WordNet suggestions.

    Args:
        lemma (str): The base form of a word (token.lemma_) to look up in WordNet.
        wn_pos (str): WordNet POS tag to filter synsets (e.g., wn.VERB, wn.ADJ, wn.ADV).

    Return:
        tuple[str]:
            The list of unique synonyms of the chosen token.
    """
    # create a set of synonyms based on the lemma and POS and ensure it is a single word
    synsets: list = wn.synsets(lemma, pos=wn_pos)
    lemmas: set = set()
    for s in synsets:
        for l in s.lemmas():
            name = l.name().replace("_", " ")
            if " " not in name:
                lemmas.add(name)
    return tuple(lemmas)


def plot_pie(dist, title, filename, stats_dir, rare_labels):
    """
    Creates and saves a pie chart and aggregates rare POS types into 'Other'.

    Args:
        dist (dict or pd.Series): distribution of POS counts or proportions
        title (str): chart title
        filename (str): file name to save the chart
        stats_dir (str): directory to save the chart
        rare_labels (list[str]): list of POS types to group into 'Other'
    """

    # define labels and sizes
    if isinstance(dist, dict):
        labels = list(dist.keys())
        sizes = list(dist.values())
    else:
        labels = dist.index
        sizes = dist.to_numpy()

    new_labels: list = []
    new_sizes: list = []
    other_size: float = 0.0

    # aggregate all the chosen labels into one
    for label, size in zip(labels, sizes):
        if label in rare_labels:
            other_size += size
        else:
            new_labels.append(label)
            new_sizes.append(size)

    if other_size > 0:
        new_labels.append("Other")
        new_sizes.append(other_size)

    # create the pie chart with the POS
    plt.figure()
    plt.pie(new_sizes, labels=new_labels, autopct="%1.1f%%")
    plt.title(title)
    plt.savefig(os.path.join(stats_dir, filename), bbox_inches="tight", dpi=300)
    plt.close()


def _normalize_counter(counter: Counter, total: int) -> dict[str, float]:
    """
    Normalizes the counter by dividing by total. Essentially turns population into probability distribution.

    Args:
        counter (Counter): Counter object with labels and values
        total (int): Sum of all the values in the Counter.

    Return:
        dict[str, float]:
            Normalized dict of values.
    """
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def aggregate_pos(column: str, df: pd.DataFrame) -> pd.Series:
    """
    Aggregates the results across the full column.

    Args:
        column (str): Name of the DataFrame column to aggregate along.
        df (pd.DataFrame): DataFrame to aggregate the data from.

    Return:
        pd.Series:
            pandas Series with all entries divided by the sum of all the entries across the column.
    """
    total: Counter = Counter()

    for entry in df[column].dropna():
        if isinstance(entry, dict):
            parsed = entry
        elif isinstance(entry, str):
            parsed = ast.literal_eval(entry)
        else:
            continue  # skip garbage safely

        total.update(parsed)

    s = pd.Series(total)
    return s / s.sum()


def augment_text(
    doc: Doc, pos_map: Mapping[str, str], replace_ratio: float, pos_tagger: Language
) -> tuple[str, Doc, dict[str, int | float | dict[str, float]]]:
    """
    Performs synonym-replacement augmentation on a parsed spaCy document.

    Args:
        doc (Doc): spaCy token object whose lemma and POS tag are used to query WordNet.
        pos_map (dict): dict of spacy POS to augment. Other types stay untouched.
        replace_ratio (float): the ratio of tokens to augment to all non-determinant and non-punctuation tokens in the sample
        pos_tagger (Language): spaCy language object, computes POS and lemmas of the tokens

    Return:
        str: augmented string of text
        Doc: POS of the augmented string of text
        dict[str, int | float | dict[str, float]]: stats about the augmented sample
    """

    # create list of words (indices) to augment
    tokens: list = list(doc)
    candidates: list = [
        i for i, t in enumerate(tokens) if t.pos_ in pos_map and t.ent_type_ == ""
    ]
    num_candidates: int = len(candidates)

    # calculate the number of non-determinant/non-punctuation tokens
    non_det_tokens: int = int(
        len([t for t in tokens if t.pos_ not in ["PUNCT", "DET"]])
    )
    k: int = min(int(non_det_tokens * replace_ratio), len(candidates))
    replaced: int = 0
    pos_before: Counter = Counter(t.pos_ for t in tokens)
    pos_before_norm: dict[str, float] = _normalize_counter(pos_before, len(tokens))
    replaced_pos_counter: Counter = Counter()

    if not candidates:
        # if there are no candidates at all, return the original sequence
        text = "".join(
            t.text_with_ws if isinstance(t, Token) else str(t) + " " for t in tokens
        ).strip()
        fallback_doc = spacy.tokens.Doc(doc.vocab, words=[t.text for t in tokens])
        return (
            text,
            fallback_doc,
            {
                "tokens_total": len(tokens),
                "tokens_candidate": 0,
                "tokens_replaced": 0,
                "replace_ratio_actual": 0.0,
                "pos_dist_before": pos_before_norm,
                "pos_dist_after": pos_before_norm,
                "replaced_ADJ": 0,
                "replaced_ADV": 0,
                "replaced_VERB": 0,
            },
        )

    # augment until either the number of augmented words reaches the threshold or number of possible candidates ends
    while k > 0 and len(candidates) > 0:
        chosen: int = random.choice(candidates)
        synonyms = cached_synonyms(
            tokens[chosen].lemma_.lower(), pos_map[tokens[chosen].pos_]
        )
        synonyms = [s for s in synonyms if s.lower() != tokens[chosen].text.lower()]
        if len(synonyms) > 0:
            replaced_pos_counter[tokens[chosen].pos_] += 1
            tokens[chosen] = random.choice(synonyms)
            replaced += 1
            k -= 1
        candidates.remove(chosen)

    # concatenate the augmented sequence and tag it with POS
    augmented_text: str = " ".join(map(str, tokens))
    augmented_doc: Doc = pos_tagger(augmented_text)

    pos_after = Counter(t.pos_ for t in augmented_doc)
    pos_after_norm = _normalize_counter(pos_after, len(augmented_doc))

    # collect and return stats
    stats = {
        "tokens_total": len(tokens),
        "tokens_candidate": num_candidates,
        "tokens_replaced": replaced,
        "replace_ratio_actual": replaced / max(1, non_det_tokens),
        "pos_dist_before": pos_before_norm,
        "pos_dist_after": pos_after_norm,
        "replaced_ADJ": replaced_pos_counter.get("ADJ", 0),
        "replaced_ADV": replaced_pos_counter.get("ADV", 0),
        "replaced_VERB": replaced_pos_counter.get("VERB", 0),
    }
    # return the augmented parsed text, POS list and stats
    text: str = spacy.tokens.Doc(doc.vocab, words=[str(t) for t in tokens]).text
    return text, augmented_doc, stats


def augment_batch(
    texts: Sequence[str],
    pos_tagger: Language,
    pos_map: Mapping[str, str],
    replace_ratio: float,
    batch_size: int,
    n_process: int,
    similarity: bool,
) -> tuple[list[str], list[Any]]:
    """
    Performs synonym-replacement augmentation on a parsed spaCy document.

    Args:
        texts (Sequence[str]): spaCy token object whose lemma and POS tag are used to query WordNet.
        pos_tagger (Language): spaCy language object, computes POS and lemmas of the tokens
        pos_map (dict): dict of spacy POS to augment. Other types stay untouched.
        replace_ratio (float): the ratio of tokens to augment to all non-determinant and non-punctuation tokens in the sample
        batch_size (int): number of samples in a mini-batch
        n_process (int): number of CPU workers

    Return:
        tuple[list[str], list[Any]]: the full augmented set and list of stats
    """
    augmented_texts: list = []
    stats_all: list = []
    # Using spaCy's pipe for multicore batch processing
    for docs in tqdm(
        pos_tagger.pipe(texts, batch_size=batch_size, n_process=n_process),
        total=len(texts),
    ):
        augmented_text, aug_doc, stats = augment_text(
            docs, pos_map, replace_ratio, pos_tagger
        )
        if similarity:
            sim = docs.similarity(aug_doc)
            stats["semantic_similarity"] = sim
        augmented_texts.append(augmented_text)
        stats_all.append(stats)
    return augmented_texts, stats_all


def main(tokenizer: str, similarity: bool):
    # ------------------------------------------------------------
    # loading the data
    # ------------------------------------------------------------

    # downloading the augmentation tools and preparing the variables
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    augmentation_ratio = 0.15
    batch_size = 32
    n_process = 8

    # loading the dataset and pos tagger
    data = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "train.csv"))
    pos_tagger = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    POS_MAP = {"VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

    # ------------------------------------------------------------
    # Augmenting and visualizing the data
    # ------------------------------------------------------------

    # augment the dataset
    aug_text, stats_text = augment_batch(
        data["cleaned_text"].tolist(),
        pos_tagger=pos_tagger,
        pos_map=POS_MAP,
        replace_ratio=augmentation_ratio,
        batch_size=batch_size,
        n_process=n_process,
        similarity=similarity,
    )
    aug_high, stats_high = augment_batch(
        data["highlights"].tolist(),
        pos_tagger=pos_tagger,
        pos_map=POS_MAP,
        replace_ratio=augmentation_ratio,
        batch_size=batch_size,
        n_process=n_process,
        similarity=similarity,
    )

    # concatenate the augmented data with the original samples
    aug_df = data.copy()
    aug_df["cleaned_text"] = aug_text
    aug_df["highlights"] = aug_high
    data = pd.concat([data, aug_df], ignore_index=True)

    # gather the data and get statistics
    stats_df = pd.DataFrame(stats_text + stats_high)
    summary = stats_df.select_dtypes("number").describe()
    pos_totals = stats_df[["replaced_ADJ", "replaced_ADV", "replaced_VERB"]].sum()

    # get directory of statistics
    STATS_DIR = os.path.join(ROOT_DIR, "data", "stats")
    os.makedirs(STATS_DIR, exist_ok=True)

    # get the numbers of augmented words for each POS
    labels = pos_totals.index.str.replace("replaced_", "")
    sizes: npt.NDArray = pos_totals.values

    # visualize via a pie chart
    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Distribution of Augmented POS")
    plt.savefig(
        os.path.join(STATS_DIR, "augmented_pos_distribution.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    if similarity:
        # analyze similarity function
        subset = stats_df.sample(n=len(data["cleaned_text"]), random_state=42)
        mean_sim = subset["semantic_similarity"].mean()

        # visualize the similarity distribution
        plt.figure()
        plt.hist(subset["semantic_similarity"], bins=50)
        plt.axvline(mean_sim, color="red")
        plt.title(f"Semantic Similarity Distribution (mean={mean_sim:.4f})")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.join(STATS_DIR, "semantic_similarity_hist.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # aggregate words by POS before and after augmentation
    pos_before: pd.Series[str | int] = aggregate_pos("pos_dist_before", stats_df)
    pos_after: pd.Series[str | int] = aggregate_pos("pos_dist_after", stats_df)

    # build pie charts with distributions
    plot_pie(
        pos_before,
        "POS Distribution Before Augmentation",
        "pos_before.png",
        STATS_DIR,
        ["SCONJ", "INTJ", "X", "SYM", "SPACE"],
    )
    plot_pie(
        pos_after,
        "POS Distribution After Augmentation",
        "pos_after.png",
        STATS_DIR,
        ["SCONJ", "INTJ", "X", "SYM", "SPACE"],
    )

    # build POS-based distribution in the form of the barplot
    df = pd.DataFrame({"Before": pos_before, "After": pos_after}).fillna(0)
    plt.figure()
    df.plot(kind="bar")
    plt.title("POS Distribution Shift")
    plt.ylabel("Proportion")
    plt.savefig(
        os.path.join(STATS_DIR, "pos_shift_bar.png"), bbox_inches="tight", dpi=300
    )
    plt.close()

    # ------------------------------------------------------------
    # Saving the data
    # ------------------------------------------------------------

    summary.to_csv(
        os.path.join(ROOT_DIR, "data", "stats", "augmentation_stats_summary.csv"),
        index=False,
    )
    data.to_csv(
        os.path.join(ROOT_DIR, "data", "processed", f"{tokenizer}_augmented.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BPE",
        choices=["BPE", "WordPiece", "Canine"],
        help="The name of the file to save into",
    )
    parser.add_argument(
        "--similarity",
        action="store_false",
        help="Whether to calculate similarity score between original and augmented text (significantly slows down the process)",
    )
    args = parser.parse_args()

    main(tokenizer="BPE", similarity=False)
