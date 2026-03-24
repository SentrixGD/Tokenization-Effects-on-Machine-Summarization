# Model Checkpoint

This folder contains the checkpoints of trained models.

**Note:** The checkpoints are large and are **not tracked by Git**.

You can download it from the Hugging Face model repo:
https://huggingface.co/SentrixGD/cnn-dm-summarization

## Programmatic download

You can download the entire `checkpoints/` folder using `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
import os
import shutil

# Download only the checkpoints folder from HF repo
hf_dir = snapshot_download(
    repo_id="SentrixGD/cnn-dm-summarization",
    allow_patterns="checkpoints/*"
)

# Local project directory
LOCAL_DIR = "checkpoints"
os.makedirs(LOCAL_DIR, exist_ok=True)

# Copy files from HF cache to local checkpoints/
src_dir = os.path.join(hf_dir, "checkpoints")

for file_name in os.listdir(src_dir):
    src_path = os.path.join(src_dir, file_name)
    dst_path = os.path.join(LOCAL_DIR, file_name)
    shutil.copy2(src_path, dst_path)

print("All checkpoints downloaded to", LOCAL_DIR)
```