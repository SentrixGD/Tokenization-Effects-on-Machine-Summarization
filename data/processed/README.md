# Processed Data

This folder contains the dataset after preprocessing (cleaning, filtering, and normalization) and augmentation (wordnet synonym replacement).

**Note:** Files in this folder are large and are **not tracked by Git**.

To reproduce this data:

1. Download the raw dataset (see `data/raw/README.md`)
2. Run the preprocessing pipeline:

```bash
python -m src.data.preprocessing
python -m src.data.augmentation
