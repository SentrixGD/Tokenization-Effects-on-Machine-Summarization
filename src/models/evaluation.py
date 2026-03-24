import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import Model
from src.tokenizers.bpe_tokenizer import BPETokenizer


# --- Loading a checkpoint ---
def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    filename: str,
) -> torch.nn.Module:
    """
    Load the saved model.

    Args:
        model (torch.nn.Module): Model to load checkpoint into.
        path (str): Directory containing the checkpoint file.
        filename (str): Filename of the checkpoint file.

    Returns:
        model (torch.nn.Module): Loaded model.
    """
    # load the checkpoint
    checkpoint_path = os.path.join(path, filename)
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    # load the model state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # return the loaded model, optimizer, epoch, loss, and step
    return model


def nucleus_sampling(
    model,
    src,
    bos_id,
    eos_id,
    pad_id,
    max_len=128,
    min_len=10,
    top_p=0.9,
    temperature=0.9,
    N=0.8,
    theta=1.0,
    device="cuda",
):
    model.eval()

    src = src.unsqueeze(0).to(device)
    seq = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for step in range(max_len):

        with torch.no_grad():
            logits = model(src, seq)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # prevent EOS too early
        if step < min_len:
            probs[eos_id] = 0.0
        # 🔹 compute entropy
        entropy = -(probs * torch.log(probs + 1e-12)).sum()
        T = temperature * (N ** (theta / (entropy + 1e-12)))  # avoid div by zero
        T = max(T, 1e-2)
        logits = logits / T
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        # 🔥 nucleus filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # keep smallest set with cumulative prob >= top_p
        cutoff = cumulative_probs > top_p
        indices_over = torch.where(cutoff)[0]
        if len(indices_over) == 0:
            cutoff_idx = len(probs) - 1  # keep all
        else:
            cutoff_idx = indices_over[0]

        filtered_probs = sorted_probs[: cutoff_idx + 1]
        filtered_indices = sorted_indices[: cutoff_idx + 1]

        # normalize
        filtered_probs = filtered_probs / filtered_probs.sum()

        # sample
        next_token = filtered_indices[torch.multinomial(filtered_probs, 1)].view(1, 1)

        seq = torch.cat([seq, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return seq.squeeze(0)


def str2list(s):
    """
    Convert a string of comma-separated integers to a list of integers.

    Args:
        s (str): String of comma-separated integers

    Returns:
        list[int]: List of integers
    """
    return np.fromstring(s[1:-1], sep=",", dtype=int).tolist()


def run_nucleus(x: torch.Tensor, model: torch.nn.Module, max_len: int, min_len: int):
    ids = x.clone().detach()
    # match training preprocessing
    ids = ids[:1182]
    ids = ids.cuda()
    return nucleus_sampling(
        model,
        ids,
        2,
        3,
        0,
        top_p=0.9,
        temperature=0.9,
        max_len=max_len,
        min_len=min_len,
    )


def dataset_from_df(df):
    data = []
    for src, tgt in zip(df["tokenized_articles"], df["tokenized_highlights"]):
        src_ids = torch.tensor(str2list(src), dtype=torch.long)[:1182]
        tgt_ids = torch.tensor(str2list(tgt), dtype=torch.long)
        data.append((src_ids, tgt_ids))
    return data


def clean(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [i for i in ids if i not in {0, 2, 3}]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch


def main(tokenizer_type: str, vocab_size: int, max_len: int, min_len: int):
    tqdm.pandas(ncols=100, dynamic_ncols=True)
    torch.backends.cudnn.benchmark = True
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    TOKENIZER_REGISTRY = {
        "BPE": BPETokenizer,
    }
    tokenizer = TOKENIZER_REGISTRY[tokenizer_type]()
    tokenizer_model_path = os.path.join(
        ROOT_DIR, "data", "tokenized", f"{tokenizer_type}_tokenizer.model"
    )
    tokenizer.load(tokenizer_model_path)
    model = load_checkpoint(
        Model(vocab_size, 768, 12, 8, 0.1, 0.1, 0, 1182).cuda(),
        CHECKPOINT_DIR,
        f"latest_{tokenizer_type}.pt",
    )
    test_data = pd.read_csv(
        os.path.join(
            ROOT_DIR, "data", "tokenized", f"tokenized_test_{tokenizer_type}.csv"
        )
    )
    dataset = dataset_from_df(test_data)

    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    predictions = []
    references = []

    for src_batch, tgt_batch in tqdm(loader):

        src_batch = src_batch.cuda(non_blocking=True)

        for src, tgt in zip(src_batch, tgt_batch):

            with torch.no_grad():
                pred = run_nucleus(src, model, max_len, min_len)  # GPU

            pred = pred.cpu()  # move once

            predictions.append(clean(pred))
            references.append(clean(tgt))
    pred_texts = [tokenizer.decode(p) for p in predictions]
    ref_texts = [tokenizer.decode(r) for r in references]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = [scorer.score(ref, pred) for ref, pred in zip(ref_texts, pred_texts)]
    rouge1 = np.mean([s["rouge1"].fmeasure for s in scores])
    rouge2 = np.mean([s["rouge2"].fmeasure for s in scores])
    rougeL = np.mean([s["rougeL"].fmeasure for s in scores])
    meteor_scores = [
        meteor_score([ref.split()], pred.split())
        for ref, pred in zip(ref_texts, pred_texts)
    ]
    P, R, F1 = bert_score(pred_texts, ref_texts, lang="en", rescale_with_baseline=True)
    P = torch.as_tensor(P)
    R = torch.as_tensor(R)
    F1 = torch.as_tensor(F1)
    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    bert_f1 = F1.mean().item()

    meteor = np.mean(meteor_scores)
    results = {
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "METEOR": meteor,
        "BERT-Precision": bert_precision,
        "BERT-Recall": bert_recall,
        "BERT-F1": bert_f1,
    }

    with open(
        os.path.join(ROOT_DIR, "data", "stats", f"metrics_{tokenizer_type}.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BPE",
        choices=["BPE", "WordPiece", "Canine"],
        help="Tokenizer regime",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size used for tokenization",
    )

    parser.add_argument(
        "--max_len", type=int, default=133, help="Maximum output sequence length"
    )

    parser.add_argument(
        "--min_len", type=int, default=20, help="Minimum output sequence length"
    )

    args = parser.parse_args()

    main(
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        min_len=args.min_len,
    )
