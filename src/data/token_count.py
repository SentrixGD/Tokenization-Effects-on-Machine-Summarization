import ast
import json
import os

import pandas as pd
from tqdm import tqdm

tqdm.pandas(dynamic_ncols=True)

ROOT_DIR: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

files = [
    os.path.join(ROOT_DIR, "data", "tokenized", "tokenized_train_BPE.csv"),
    os.path.join(ROOT_DIR, "data", "tokenized", "tokenized_train_Word.csv"),
    os.path.join(ROOT_DIR, "data", "tokenized", "tokenized_train_Char.csv"),
    os.path.join(ROOT_DIR, "data", "tokenized", "tokenized_train_Unigram.csv"),
]
results = {}

for file in files:
    df = pd.read_csv(file)
    progress_bar = tqdm(range(len(df)), desc=f"Processing {file}"[-15:])

    oov_count = 0
    tokens_count = 0

    for i in progress_bar:
        # Convert string to list
        tokens_articles = ast.literal_eval(df["tokenized_articles"][i])
        tokens_highlights = ast.literal_eval(df["tokenized_highlights"][i])

        # Count OOV (token == 1) and total tokens
        oov_count += tokens_articles.count(1) + tokens_highlights.count(1)
        tokens_count += len(tokens_articles) + len(tokens_highlights)

        progress_bar.set_postfix(
            oov_count=oov_count,
            tokens_count=tokens_count,
            ratio=oov_count / tokens_count,
        )

    print(f"{file}: tokens={tokens_count}, OOV={oov_count}")

    # Store results in dict
    results[file] = {
        "tokens_count": tokens_count,
        "oov_count": oov_count,
        "oov_ratio": oov_count / tokens_count,
    }

# Write all results to one JSON file
with open(os.path.join(ROOT_DIR, "data", "stats", "tokens_oov_summary.json"), "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to tokens_oov_summary.json")
