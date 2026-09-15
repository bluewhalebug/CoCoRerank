import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import wandb

@dataclass
class Config:
    model_name: str = os.environ["MODEL_NAME"]
    train_path: str = os.environ["TRAIN_PATH"]
    val_path: str = os.environ["VAL_PATH"]
    save_path: str = os.environ["SAVE_PATH"]
    batch_size: int = int(os.environ.get("BATCH_SIZE", "96"))
    commit_max_len: int = int(os.environ.get("COMMIT_MAX_LEN", "2048"))
    meta_max_len: int = int(os.environ.get("META_MAX_LEN", "32"))
    subject_max_len: int = int(os.environ.get("SUBJECT_MAX_LEN", "64"))
    lr: float = float(os.environ.get("LR", "2e-5"))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    epochs: int = int(os.environ.get("EPOCHS", "8"))
    temperature: float = float(os.environ.get("TEMP", "0.05"))
    pooling: str = (
        os.environ.get("POOLING", "mean").strip().lower().replace("-", "_")
    )
    type_loss_weight: float = float(os.environ.get("TYPE_LOSS_WEIGHT", "0"))
    warmup_ratio: float = float(os.environ.get("WARMUP_RATIO", "0.06"))
    max_grad_norm: float = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
    num_workers: int = int(os.environ.get("NUM_WORKERS", "4"))
    seed: int = int(os.environ.get("SEED", "42"))
    device: str = os.environ.get("DEVICE", "cuda:0")
    wandb_project: str = os.environ.get("WANDB_PROJECT", "commit-contrastive")
    wandb_run_name: str = os.environ.get("WANDB_RUN_NAME", "formal_easy_model")
    wandb_dir: str = os.environ["WANDB_DIR"]

    def __post_init__(self):
        if self.pooling == "last_token":
            self.pooling = "last"
        if self.pooling not in {"mean", "last"}:
            raise ValueError("POOLING must be one of: 'mean', 'last', 'last_token'")


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = CFG.seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def init_wandb() -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    Path(CFG.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb.init(
        project=CFG.wandb_project,
        name=CFG.wandb_run_name,
        mode=os.environ.get("WANDB_MODE", "offline"),
        dir=CFG.wandb_dir,
        config=asdict(CFG),
        settings=wandb.Settings(init_timeout=180),
    )


def read_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                yield json.loads(line)


def build_type_vocab(*paths: str) -> dict[str, int]:
    values = set()
    for path in paths:
        for obj in read_jsonl(path):
            values.add(normalize_field(obj.get("type"), "<none>"))
    return {value: idx for idx, value in enumerate(sorted(values))}


def normalize_field(value, default: str = "<none>") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


class CommitDataset(Dataset):
    def __init__(self, path: str, type_to_id: dict[str, int]):
        self.data = []
        type_counter = Counter()

        for obj in read_jsonl(path):
            commit = normalize_field(obj.get("commit"), "")
            commit_type = normalize_field(obj.get("type"), "<none>")
            scope = normalize_field(obj.get("scope"), "<none>")
            subject = normalize_field(obj.get("subject"), "")
            if not commit or not subject:
                continue

            type_counter[commit_type] += 1
            meta = f"type: {commit_type}\nscope: {scope}"
            self.data.append(
                {
                    "commit": commit,
                    "meta": meta,
                    "subject": subject,
                    "type_id": type_to_id[commit_type],
                }
            )

        if not self.data:
            raise ValueError(f"{path} has no usable samples")

        print(f"Loaded {len(self.data)} samples from {path}")
        print("Type distribution:", dict(type_counter.most_common()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        commits = [item["commit"] for item in batch]
        metas = [item["meta"] for item in batch]
        subjects = [item["subject"] for item in batch]
        type_ids = torch.tensor([item["type_id"] for item in batch], dtype=torch.long)

        commit_enc = self.tokenizer(
            commits,
            padding=True,
            truncation=True,
            max_length=CFG.commit_max_len,
            return_tensors="pt",
        )
        meta_enc = self.tokenizer(
            metas,
            padding=True,
            truncation=True,
            max_length=CFG.meta_max_len,
            return_tensors="pt",
        )
        subject_enc = self.tokenizer(
            subjects,
            padding=True,
            truncation=True,
            max_length=CFG.subject_max_len,
            return_tensors="pt",
        )

        return commit_enc, meta_enc, subject_enc, type_ids, subjects


class StructuredBiEncoder(nn.Module):
    def __init__(self, model_name: str, num_types: int, pooling: str):
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be either 'mean' or 'last'")
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.type_classifier = nn.Linear(hidden_size, num_types)

    @staticmethod
    def mean_pool(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1)
        return (hidden * mask).sum(dim=1) / denom

    @staticmethod
    def last_token_pool(hidden, attention_mask):
        token_positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(
            0
        )
        last_token_positions = token_positions.masked_fill(attention_mask == 0, -1)
        lengths = last_token_positions.argmax(dim=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]

    def encode_text(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            return self.mean_pool(outputs.last_hidden_state, attention_mask)
        return self.last_token_pool(outputs.last_hidden_state, attention_mask)

    def encode_context(self, commit_ids, commit_mask, meta_ids, meta_mask):
        commit_emb = self.encode_text(commit_ids, commit_mask)
        meta_emb = self.encode_text(meta_ids, meta_mask)
        ctx_emb = self.context_projection(torch.cat([commit_emb, meta_emb], dim=-1))
        ctx_emb = nn.functional.normalize(ctx_emb, dim=-1)
        type_logits = self.type_classifier(ctx_emb)
        return ctx_emb, type_logits

    def encode_subject(self, sub_ids, sub_mask):
        sub_emb = self.encode_text(sub_ids, sub_mask)
        return nn.functional.normalize(sub_emb, dim=-1)

    def forward(self, commit_enc, meta_enc, subject_enc):
        ctx_emb, type_logits = self.encode_context(
            commit_enc["input_ids"],
            commit_enc["attention_mask"],
            meta_enc["input_ids"],
            meta_enc["attention_mask"],
        )
        sub_emb = self.encode_subject(
            subject_enc["input_ids"],
            subject_enc["attention_mask"],
        )
        return ctx_emb, sub_emb, type_logits


def multi_positive_ce(scores, keys):
    key_tensor = torch.tensor(keys, device=scores.device)
    positive_mask = key_tensor.unsqueeze(0).eq(key_tensor.unsqueeze(1))
    log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    positive_log_probs = log_probs.masked_fill(~positive_mask, -torch.inf)
    return -torch.logsumexp(positive_log_probs, dim=1).mean()


def batch_accuracy(scores, keys):
    pred = scores.argmax(dim=1).detach().cpu()
    return sum(keys[i] == keys[pred_i] for i, pred_i in enumerate(pred.tolist())) / len(keys)


def compute_loss_and_metrics(ctx_emb, sub_emb, type_logits, type_ids, subject_keys):
    scores = torch.matmul(ctx_emb, sub_emb.t()) / CFG.temperature
    key_to_id = {key: idx for idx, key in enumerate(sorted(set(subject_keys)))}
    keys = [key_to_id[key] for key in subject_keys]

    loss_ctx = multi_positive_ce(scores, keys)
    loss_sub = multi_positive_ce(scores.t(), keys)
    contrast_loss = (loss_ctx + loss_sub) / 2
    type_loss = nn.CrossEntropyLoss()(type_logits, type_ids)
    loss = contrast_loss + CFG.type_loss_weight * type_loss

    return {
        "loss": loss,
        "contrast_loss": contrast_loss.detach(),
        "type_loss": type_loss.detach(),
        "ctx_acc": batch_accuracy(scores, keys),
        "sub_acc": batch_accuracy(scores.t(), keys),
        "type_acc": (type_logits.argmax(dim=1) == type_ids).float().mean().detach(),
    }


def move_batch(commit_enc, meta_enc, subject_enc, type_ids, device):
    commit_enc = {key: value.to(device, non_blocking=True) for key, value in commit_enc.items()}
    meta_enc = {key: value.to(device, non_blocking=True) for key, value in meta_enc.items()}
    subject_enc = {key: value.to(device, non_blocking=True) for key, value in subject_enc.items()}
    type_ids = type_ids.to(device, non_blocking=True)
    return commit_enc, meta_enc, subject_enc, type_ids


def run_epoch(model, loader, optimizer, scheduler, scaler, amp_device, train_mode, global_step):
    model.train(train_mode)
    totals = Counter()
    progress = tqdm(loader, desc="Train" if train_mode else "Val")

    for commit_enc, meta_enc, subject_enc, type_ids, subject_keys in progress:
        commit_enc, meta_enc, subject_enc, type_ids = move_batch(
            commit_enc, meta_enc, subject_enc, type_ids, CFG.device
        )

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
                ctx_emb, sub_emb, type_logits = model(commit_enc, meta_enc, subject_enc)
                outputs = compute_loss_and_metrics(
                    ctx_emb, sub_emb, type_logits, type_ids, subject_keys
                )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(outputs["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                wandb.log(
                    {
                        "train/loss": outputs["loss"].item(),
                        "train/contrast_loss": outputs["contrast_loss"].item(),
                        "train/type_loss": outputs["type_loss"].item(),
                        "train/ctx_acc": outputs["ctx_acc"],
                        "train/sub_acc": outputs["sub_acc"],
                        "train/type_acc": outputs["type_acc"].item(),
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                global_step += 1

        totals["loss"] += outputs["loss"].item()
        totals["contrast_loss"] += outputs["contrast_loss"].item()
        totals["type_loss"] += outputs["type_loss"].item()
        totals["ctx_acc"] += outputs["ctx_acc"]
        totals["sub_acc"] += outputs["sub_acc"]
        totals["type_acc"] += outputs["type_acc"].item()

    denom = max(len(loader), 1)
    metrics = {key: value / denom for key, value in totals.items()}
    return metrics, global_step


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, type_to_id):
    save_path = Path(CFG.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": asdict(CFG),
            "type_to_id": type_to_id,
        },
        save_path,
    )


def train():
    set_seed(CFG.seed)
    init_wandb()

    type_to_id = build_type_vocab(CFG.train_path, CFG.val_path)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CommitDataset(CFG.train_path, type_to_id)
    val_dataset = CommitDataset(CFG.val_path, type_to_id)
    collator = Collator(tokenizer)
    generator = torch.Generator().manual_seed(CFG.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = StructuredBiEncoder(CFG.model_name, len(type_to_id), CFG.pooling).to(
        CFG.device
    )
    model.encoder.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    total_steps = max(1, len(train_loader) * CFG.epochs)
    warmup_steps = math.ceil(total_steps * CFG.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    amp_device = "cuda" if CFG.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=(amp_device == "cuda"))

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(CFG.epochs):
        train_metrics, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, scaler, amp_device, True, global_step
        )
        val_metrics, global_step = run_epoch(
            model, val_loader, optimizer, scheduler, scaler, amp_device, False, global_step
        )

        print(f"\nEpoch {epoch + 1}")
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")
        wandb.log(
            {
                "epoch": epoch,
                **{f"train_epoch/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, type_to_id)
            wandb.run.summary["best_val_loss"] = best_val_loss
            print(f"Saved best model to {CFG.save_path}")

    wandb.finish()


if __name__ == "__main__":
    train()
