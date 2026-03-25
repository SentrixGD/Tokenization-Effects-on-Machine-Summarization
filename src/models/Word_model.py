"""
Script: Word_model.py
Purpose: Define a PyTorch model for training on the Text Summarization dataset.
Inputs: Tokenized training data
Outputs: Trained model
Dependencies: torch, sentencepiece, pandas, numpy, tqdm
"""

import csv
import math
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from model import Model, init_weights
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def str2list(s):
    """
    Convert a string of comma-separated integers to a list of integers.

    Args:
        s (str): String of comma-separated integers

    Returns:
        list[int]: List of integers
    """
    return np.fromstring(s[1:-1], sep=",", dtype=int).tolist()


class TextSummaryDataset(Dataset):
    """
    A PyTorch Dataset for text summarization.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        src_col: str = "tokenized_articles",
        tgt_col: str = "tokenized_highlights",
    ):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            src_col (str): Column name for source text
            tgt_col (str): Column name for target text
        """
        self.src = df[src_col].tolist()
        self.tgt = df[tgt_col].tolist()

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.src)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the item at the given index.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Item at the given index
        """
        # return as torch tensors
        src_tensor = torch.tensor(self.src[idx], dtype=torch.long)
        tgt_tensor = torch.tensor(self.tgt[idx], dtype=torch.long)
        return src_tensor, tgt_tensor


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], max_src_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the Text Summarization dataset.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of tuples containing the source and target tensors.
        max_src_len (int): Maximum source sequence length.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded source and target tensors.
    """
    src_batch, tgt_batch = zip(*batch)

    # truncate the source sequences first
    src_batch = [seq[:max_src_len] for seq in src_batch]
    # no need to truncate the target sequences

    # pad the sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(list(tgt_batch), batch_first=True, padding_value=0)

    return src_padded, tgt_padded


# --- Saving a checkpoint ---
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    filename: str = "latest_Word.pt",
) -> None:
    """
    Save a checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler to save.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler to save.
        epoch (int): Current epoch.
        step (int): Current step.
        loss (float): Current loss.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename of the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    # save the checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))


# --- Loading a checkpoint ---
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    path: str,
    filename: str,
) -> Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
    torch.amp.GradScaler,
    int,
    int,
]:
    """
    Load a saved checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): Model to load checkpoint into.
        optimizer (torch.optim.Optimizer): Optimizer to load checkpoint into.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler to load checkpoint into
        path (str): Directory containing the checkpoint file.
        filename (str): Filename of the checkpoint file.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Scheduler to load checkpoint into

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, torch.amp.GradScaler, int, int]: Loaded model, optimizer, scheduler, scaler, epoch, and step.
    """
    # load the checkpoint
    checkpoint_path = os.path.join(path, filename)
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    # load the model state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # load the optimizer state dict
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    # get the epoch, loss, and step from the checkpoint
    epoch = checkpoint["epoch"]
    step = checkpoint["step"]

    # return the loaded model, optimizer, epoch, loss, and step
    return model, optimizer, scheduler, scaler, epoch, step


def lr_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    """
    Compute the learning rate lambda based on the step, total steps, and warmup steps.

    If the step is less than the warmup steps, the lambda is linearly increased
    from 0 to 1. Otherwise, the lambda is linearly decreased from 1 to 0.

    Args:
        step (int): Current step.
        total_steps (int): Total number of steps.
        warmup_steps (int): Number of warmup steps.

    Returns:
        float: Learning rate lambda.
    """
    if step < warmup_steps:
        return step / warmup_steps
    return max(0, (total_steps - step) / (total_steps - warmup_steps))


def decode_tokens(token_ids: torch.Tensor, sp: spm.SentencePieceProcessor) -> str:
    """
    Decode token IDs back into text.

    Args:
        token_ids (torch.Tensor): List of token IDs to decode.
        sp (spm.SentencePieceProcessor): Loaded SentencePiece tokenizer.

    Returns:
        str: Decoded text.
    """
    # convert tensor → list
    token_ids_list = token_ids.tolist()
    # filter out the padding token
    filtered_token_ids = [tok for tok in token_ids_list if tok != 0]
    # decode the filtered token IDs using SentencePiece
    decoded_text = sp.DecodeIds(filtered_token_ids)
    return decoded_text


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def main():
    """
    Main function for training the Text Summarization model.

    Reads the tokenized train, validation, and test datasets.
    Loads the SentencePiece tokenizer model.
    Constructs the train and validation data loaders.
    Trains the model using the Adam optimizer and a learning rate scheduler.
    Evaluates the model on the validation set after every 1000 steps.
    Saves the model's state after every 1000 steps.
    Writes the training and validation loss, learning rate, and VRAM usage to a CSV log file.
    """

    # set the seed for random events
    global SEED
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # create a torch generator
    g = torch.Generator()
    g.manual_seed(SEED)

    # --- Loading data ---
    tqdm.pandas(ncols=100, dynamic_ncols=True)

    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # load tokenized train, validation, and test datasets
    train_data = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "tokenized", "tokenized_train_Word.csv")
    )

    # load the SentencePiece tokenizer model
    model_path = os.path.join(ROOT_DIR, "data", "tokenized", "Word_tokenizer.model")
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    # define the columns to process
    cols = ["tokenized_articles", "tokenized_highlights"]

    # process the columns
    for col in cols:
        train_data[col] = train_data[col].progress_apply(str2list)

    # create TextSummaryDataset instances
    train_dataset = TextSummaryDataset(train_data)

    # set Hyperparameters
    warmup_steps = 4000
    vocab_size = 32000
    true_batch_size = 32
    batch_size = 2
    accumulation = true_batch_size // batch_size

    model = Model(vocab_size, 768, 12, 8, 0.1, 0.1, 0, 1182).float().cuda()

    # initialize model weights
    init_weights(model)

    # set loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    scaler = torch.amp.GradScaler(
        device="cuda",
        init_scale=2.0**10,  # big enough to start safely
        growth_factor=2.0,  # aggressive growth
        backoff_factor=0.5,  # shrink on overflow
        growth_interval=2000,  # only increase after enough safe steps
    )
    # get total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    tokens_per_sample = 62.36000463660601 + 684.7930636374174

    # create optimizer instance
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    # set approximate steps
    approximate_steps = math.ceil(
        trainable_params * 20 // (tokens_per_sample * true_batch_size)
    )
    # create learning rate scheduler
    scheduler = LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, approximate_steps, warmup_steps),
    )
    # check if checkpoint exists
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "latest_Word.pt")):
        model, optimizer, scheduler, scaler, start_epoch, global_step = load_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            CHECKPOINT_DIR,
            "latest_Word.pt",
        )
        print(f"Resuming from epoch {start_epoch}, global_step {global_step}")
    else:
        start_epoch = 0
        global_step = 0

    # set optimal tokens
    optimal_tokens = int(trainable_params * 20) - int(
        tokens_per_sample * true_batch_size * global_step
    )
    print(
        optimal_tokens,
        optimal_tokens / (tokens_per_sample * true_batch_size),
        optimal_tokens / (tokens_per_sample * 215675 * 2),
    )

    # create DataLoaders for train and validation sets
    train_loader = DataLoader(
        train_dataset,
        batch_size=true_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, max_src_len=1182),
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        pin_memory=True,
    )

    # create deque to store the last 100 losses
    loss_window = deque(maxlen=100)

    # create log file
    log_path = os.path.join(ROOT_DIR, "runs", "Word", "log.csv")
    file_exists = os.path.isfile(log_path)
    csv_file = open(log_path, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["step", "loss", "lr", "vram_gb"])

    epoch = math.ceil(global_step / len(train_loader)) + 1
    # loop through the epochs
    while optimal_tokens > 0:
        # create tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        # loop through the training data
        for step, (src_big_batch, tgt_big_batch) in enumerate(progress_bar):

            if step < global_step % len(train_loader):
                continue

            # move the data to the GPU
            src_big_batch = src_big_batch.cuda()
            tgt_big_batch = tgt_big_batch.cuda()
            B = src_big_batch.size(0)
            micro_B = math.ceil(B / accumulation)

            # zero the gradients
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0

            found_inf = False

            for i in range(0, B, micro_B):
                src_batch = src_big_batch[i : i + micro_B]
                tgt_batch = tgt_big_batch[i : i + micro_B]

                decoder_input = tgt_batch[:, :-1]
                targets = tgt_batch[:, 1:]

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(src_batch, decoder_input)
                    if not torch.isfinite(logits).all():
                        print("Invalid logits at step", global_step)
                        found_inf = True
                        break
                    loss = (
                        loss_fn(logits.view(-1, vocab_size), targets.reshape(-1))
                        / accumulation
                    )

                if not torch.isfinite(loss):
                    print("Invalid loss at step", global_step)
                    scaler.update()
                    found_inf = True
                    break

                batch_loss += loss.item()
                scaler.scale(loss).backward()

            if found_inf:
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.unscale_(optimizer)

            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("Invalid gradients at step", global_step)
                    found_inf = True
                    break

            # gradient clipping + optimizer update
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if found_inf:
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimal_tokens -= true_batch_size * tokens_per_sample
            # append the loss to the loss window
            loss_window.append(batch_loss)
            # save the model every 1000 steps
            if (global_step + 1) % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    # get the first batch in the loader
                    src_single = src_batch[0:1]
                    tgt_single = tgt_batch[0:1]
                    # prepare the decoder input
                    decoder_input = tgt_single[:, :-1]
                    # compute the logits
                    logits = model(src_single, decoder_input)
                    # get the predicted tokens
                    pred_tokens = torch.argmax(logits, dim=-1)  # (1, T)
                    # decode the predicted tokens into text
                    pred_text = decode_tokens(pred_tokens, sp)
                    print("Teacher forced prediction:", pred_text)
                    # start with the first token in tgt (or a special <BOS>)
                    input_tok = torch.full(
                        (1, 1), 2, device=src_single.device
                    ).cuda()  # shape [1, 1]
                    outputs = []

                    # generate sequence using greedy decoding
                    for t in range(117):
                        logits = model(
                            src_single, input_tok
                        )  # (1, seq_len, vocab_size)
                        probs = torch.softmax(logits[:, -1, :] / 0.9, dim=-1)
                        next_tok = torch.argmax(probs, dim=-1, keepdim=True)
                        outputs.append(next_tok)

                        input_tok = torch.cat(
                            [input_tok, next_tok], dim=1
                        )  # append next token

                        # stop if all sequences have generated EOS
                        if (next_tok == 3).all():
                            break
                    # combine generated tokens
                    preds = torch.cat(outputs, dim=1)  # shape [1, seq_len-1]
                    pred_text = decode_tokens(preds, sp)
                    print("Predicted greedy:", pred_text)

                    # generate sequence using multinomial sampling
                    input_tok = torch.full((1, 1), 2, device=src_single.device)  # BOS
                    outputs = []
                    for t in range(117):
                        logits = model(
                            src_single, input_tok
                        )  # (1, seq_len, vocab_size)
                        probs = torch.softmax(logits[:, -1, :] / 0.9, dim=-1)
                        next_tok = torch.multinomial(probs, 1)
                        outputs.append(next_tok)

                        input_tok = torch.cat(
                            [input_tok, next_tok], dim=1
                        )  # append next token

                        # stop if all sequences have generated EOS
                        if (next_tok == 3).all():
                            break
                    # combine generated tokens
                    preds = torch.cat(outputs, dim=1)  # shape [1, seq_len-1]
                    pred_text = decode_tokens(preds, sp)
                    target_text = decode_tokens(tgt_single, sp)
                    print("Predicted multinomial:", pred_text)
                    print("Target   :", target_text)
                model.train()
                torch.cuda.empty_cache()
            # calculate the moving average of the loss
            if len(loss_window) != 0:
                avg_loss = sum(loss_window) / len(loss_window)
            else:
                avg_loss = 0
            # update the progress bar with the loss, moving average, learning rate, and remaining tokens
            progress_bar.set_postfix(
                loss=batch_loss,
                loss_moving_avg=avg_loss,
                lr=optimizer.param_groups[0]["lr"],
                tokens_left=optimal_tokens,
            )
            # write the loss, moving average, learning rate, and memory usage to the CSV file
            writer.writerow(
                [
                    global_step,
                    avg_loss,
                    optimizer.param_groups[0]["lr"],
                    torch.cuda.memory_allocated() / 1e9,
                ]
            )
            # flush the CSV file to ensure that the data is written to disk
            csv_file.flush()
            # save a checkpoint every 1000 steps, and also when the remaining tokens reach zero
            if (global_step + 1) % 1000 == 0 or optimal_tokens <= 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    CHECKPOINT_DIR,
                    filename=f"latest_Word.pt",
                )
                # if the remaining tokens reach zero, exit the training loop
                if optimal_tokens <= 0:
                    return
            # increment the global step counter
            global_step += 1
        epoch += 1


if __name__ == "__main__":
    main()
