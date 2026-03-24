# Tokenized Data

This folder contains the dataset after tokenization.

**Note:** Files in this folder are large and are **not tracked by Git**.

To reproduce this data:

1. Download the raw dataset (see `data/raw/README.md`)
2. Run the preprocessing pipeline:

```bash
python -m src.data.preprocessing
python -m src.data.augmentation
```

3. Run the tokenization pipeline:

```bash
python -m src.data.tokenization
```
Available tokenizer regimes
BPE (default) — Byte Pair Encoding
WordPiece — WordPiece tokenizer
Canine - Canine tokenizer
You can also create a new JSON file for tokenized data:

```bash
python -m src.data.tokenization --tokenizer BPE --new_json
```
