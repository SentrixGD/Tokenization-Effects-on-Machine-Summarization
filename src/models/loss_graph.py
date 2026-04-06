import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main(tokenizer: str):
    DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "runs",
        tokenizer,
        "log.csv",
    )
    SAVE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    print(DIR)

    data = pd.read_csv(DIR)

    # Keep last occurrence of each step
    data = data.drop_duplicates(subset="step", keep="last")

    # Sort after deduplication
    data = data.sort_values("step")

    # Plot
    plt.figure()
    plt.plot(data["step"], data["loss"])
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("BPE Model loss")
    plt.savefig(os.path.join(SAVE_DIR, "data", "stats", f"loss_{tokenizer}.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss")

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BPE",
        choices=["BPE", "Word", "Char", "Unigram", "Charformer"],
        help="Tokenizer regime",
    )

    args = parser.parse_args()

    main(tokenizer=args.tokenizer)
