"""Inference helpers for checkpoints produced by train/easytrain.py and hardtrain.py."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


DEFAULT_COMMIT_MAX_LEN = 2048
DEFAULT_META_MAX_LEN = 32
DEFAULT_SUBJECT_MAX_LEN = 64
DEFAULT_TEMPERATURE = 0.03


def normalize_pooling(pooling):
    pooling = (pooling or "mean").strip().lower().replace("-", "_")
    if pooling == "last_token":
        pooling = "last"
    if pooling not in {"mean", "last"}:
        raise ValueError("pooling must be one of: 'mean', 'last', 'last_token'")
    return pooling


class StructuredBiEncoder(nn.Module):
    """Bi-encoder architecture shared by training and candidate scoring."""

    def __init__(self, model_name: str, num_types: int, pooling: str):
        super().__init__()
        self.pooling = normalize_pooling(pooling)
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
        token_positions = torch.arange(
            hidden.size(1), device=hidden.device
        ).unsqueeze(0)
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
        ctx_emb = self.context_projection(
            torch.cat([commit_emb, meta_emb], dim=-1)
        )
        ctx_emb = nn.functional.normalize(ctx_emb, dim=-1)
        type_logits = self.type_classifier(ctx_emb)
        return ctx_emb, type_logits

    def encode_subject(self, sub_ids, sub_mask):
        sub_emb = self.encode_text(sub_ids, sub_mask)
        return nn.functional.normalize(sub_emb, dim=-1)


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        checkpoint = {"model_state_dict": state_dict, "config": {}}
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    return checkpoint, state_dict


def infer_num_types(checkpoint, state_dict):
    type_to_id = checkpoint.get("type_to_id")
    if type_to_id:
        return len(type_to_id)
    classifier_weight = state_dict.get("type_classifier.weight")
    if classifier_weight is None:
        raise KeyError(
            "Cannot infer num_types: missing type_to_id and type_classifier.weight"
        )
    return classifier_weight.shape[0]


def resolve_config(args, checkpoint):
    config = checkpoint.get("config") or {}
    model_name = config.get("model_name") or args.model_name
    if not model_name:
        raise ValueError(
            "model_name is not set. Pass --model-name or use a checkpoint with config."
        )
    commit_max_len = int(
        config.get("commit_max_len")
        or args.commit_max_len
        or DEFAULT_COMMIT_MAX_LEN
    )
    meta_max_len = int(
        config.get("meta_max_len") or args.meta_max_len or DEFAULT_META_MAX_LEN
    )
    subject_max_len = int(
        config.get("subject_max_len")
        or args.subject_max_len
        or DEFAULT_SUBJECT_MAX_LEN
    )
    pooling = normalize_pooling(config.get("pooling") or args.pooling or "mean")
    score_temperature = config.get("temperature")
    if score_temperature is None:
        score_temperature = args.score_temperature
    if score_temperature is None:
        score_temperature = DEFAULT_TEMPERATURE
    return (
        model_name,
        commit_max_len,
        meta_max_len,
        subject_max_len,
        pooling,
        float(score_temperature),
    )


def get_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def get_amp_dtype(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def tokenize_to_device(tokenizer, texts, max_length, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        key: value.to(device, non_blocking=True) for key, value in encoded.items()
    }


def encode_subjects(
    model,
    tokenizer,
    subjects,
    max_length,
    subject_batch_size,
    device,
    amp_dtype,
):
    chunks = []
    for start in range(0, len(subjects), subject_batch_size):
        subject_enc = tokenize_to_device(
            tokenizer,
            subjects[start : start + subject_batch_size],
            max_length,
            device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            sub_emb = model.encode_subject(
                subject_enc["input_ids"],
                subject_enc["attention_mask"],
            )
        chunks.append(sub_emb)
    return torch.cat(chunks, dim=0)
