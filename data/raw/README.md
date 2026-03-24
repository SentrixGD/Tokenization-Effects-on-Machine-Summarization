# Raw Data

This folder contains the raw input data used in the project.

**Note:** The dataset is large and is **not tracked by Git**.

You can download it from the Hugging Face model repo:
https://huggingface.co/SentrixGD/cnn-dm-summarization

## Expected structure

After downloading, the `raw/` folder in the Hugging Face repo should contain:

raw/
├── train.csv
├── validation.csv
└── test.csv

---

## Programmatic download

You can download all three splits using the `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download
import os

# Map Hugging Face raw/ folder to project data/raw/
PROJECT_RAW_DIR = "data/raw"
os.makedirs(PROJECT_RAW_DIR, exist_ok=True)

for split in ["train", "validation", "test"]:
    file_path = hf_hub_download(
        repo_id="SentrixGD/cnn-dm-summarization",
        filename=f"raw/{split}.csv"
    )
    # Copy to full project raw folder
    dest_path = os.path.join(PROJECT_RAW_DIR, f"{split}.csv")
    with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())

print("All raw data files downloaded to", PROJECT_RAW_DIR)
