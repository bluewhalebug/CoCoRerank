import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class MeanPoolEncoder(nn.Module):
    def __init__(self, model_path, torch_dtype=None, trust_remote_code=False):
        super().__init__()
        kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.encoder = AutoModel.from_pretrained(model_path, **kwargs)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(emb, dim=-1)


class Collator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode(self, texts):
        kwargs = {
            "padding": True,
            "return_tensors": "pt",
        }

        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length
        else:
            kwargs["truncation"] = False

        return self.tokenizer(list(texts), **kwargs)

    def __call__(self, batch):
        commits = [item["commit"] for item in batch]
        subjects = [item["subject"] for item in batch]

        commit_enc = self._encode(commits)
        subject_enc = self._encode(subjects)

        meta = [
            {
                "input_line_no": item["input_line_no"],
                "line_no": item["line_no"],
                "sha": item["sha"],
                "type": item["type"],
                "scope": item["scope"],
                "subject": item["subject"],
            }
            for item in batch
        ]

        return commit_enc, subject_enc, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use embedding cosine similarity between commit and subject to find suspicious noisy samples."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-length",
        type=int,
        default=131072,
        help="Token max length. Default uses the model maximum length to avoid position index errors.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate tokenizer inputs. This can be very slow or fail on long commits.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, samples with score <= threshold are written as suspicious.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.05,
        help="Write the lowest-scoring ratio as suspicious when threshold is not set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Write the lowest-scoring K samples as suspicious. Overrides top-ratio when set.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 even on CUDA. Default uses fp16 on CUDA to reduce memory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write failed batches to embedding_errors.jsonl and continue.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_max_length(tokenizer, max_length_arg, no_truncate):
    if no_truncate:
        return None

    if max_length_arg is not None:
        if max_length_arg <= 0:
            raise ValueError("--max-length must be positive")
        return max_length_arg

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None or model_max_length > 10**8:
        return None
    return int(model_max_length)


def iter_jsonl_batches(path, batch_size):
    batch = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)
            batch.append(
                {
                    "input_line_no": line_no,
                    "line_no": obj.get("original_line_no", obj.get("line_no", line_no)),
                    "sha": obj.get("sha"),
                    "type": obj.get("type"),
                    "scope": obj.get("scope"),
                    "subject": obj.get("subject", ""),
                    "commit": obj.get("commit", ""),
                }
            )

            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def main():
    args = parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"{args.input_path} not found")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if not 0 <= args.top_ratio <= 1:
        raise ValueError("--top-ratio must be in [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    torch_dtype = None
    if device.startswith("cuda") and not args.fp32:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = resolve_max_length(tokenizer, args.max_length, args.no_truncate)
    collator = Collator(tokenizer, max_length=max_length)

    model = MeanPoolEncoder(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    scored_path = output_dir / "embedding_scores.jsonl"
    errors_path = output_dir / "embedding_errors.jsonl"
    all_scores = []
    total = 0
    errors = 0

    with (
        open(scored_path, "w", encoding="utf-8") as fout,
        open(errors_path, "w", encoding="utf-8") as ferr,
        torch.no_grad(),
    ):
        batches = iter_jsonl_batches(args.input_path, args.batch_size)
        for batch in tqdm(batches, desc="Scoring"):
            total += len(batch)
            try:
                commit_enc, subject_enc, meta = collator(batch)
                commit_ids = commit_enc["input_ids"].to(device)
                commit_mask = commit_enc["attention_mask"].to(device)
                subject_ids = subject_enc["input_ids"].to(device)
                subject_mask = subject_enc["attention_mask"].to(device)

                commit_emb = model.encode(commit_ids, commit_mask)
                subject_emb = model.encode(subject_ids, subject_mask)
                scores = F.cosine_similarity(commit_emb, subject_emb, dim=-1)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                errors += len(batch)
                for item in batch:
                    error_item = {
                        "input_line_no": item["input_line_no"],
                        "line_no": item["line_no"],
                        "sha": item["sha"],
                        "type": item["type"],
                        "scope": item["scope"],
                        "subject": item["subject"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    ferr.write(json.dumps(error_item, ensure_ascii=False) + "\n")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

            for item, score in zip(meta, scores.detach().cpu().tolist()):
                score_item = {
                    "input_line_no": item["input_line_no"],
                    "line_no": item["line_no"],
                    "sha": item["sha"],
                    "type": item["type"],
                    "scope": item["scope"],
                    "subject": item["subject"],
                    "embedding_similarity": score,
                }
                all_scores.append(score_item)
                fout.write(json.dumps(score_item, ensure_ascii=False) + "\n")

    sorted_scores = sorted(all_scores, key=lambda x: x["embedding_similarity"])

    if args.threshold is not None:
        suspicious_scores = [
            item
            for item in sorted_scores
            if item["embedding_similarity"] <= args.threshold
        ]
    elif args.top_k is not None:
        suspicious_scores = sorted_scores[: max(args.top_k, 0)]
    else:
        top_k = int(len(sorted_scores) * args.top_ratio)
        suspicious_scores = sorted_scores[: max(top_k, 1)]

    suspicious_input_line_nos = {item["input_line_no"] for item in suspicious_scores}
    score_by_input_line_no = {
        item["input_line_no"]: item["embedding_similarity"] for item in suspicious_scores
    }
    original_line_no_by_input_line_no = {
        item["input_line_no"]: item["line_no"] for item in suspicious_scores
    }

    suspicious_path = output_dir / "suspicious_embedding_noise.jsonl"
    with open(suspicious_path, "w", encoding="utf-8") as fout:
        with open(args.input_path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                if line_no not in suspicious_input_line_nos:
                    continue
                raw = json.loads(line)
                raw["embedding_similarity"] = score_by_input_line_no[line_no]
                raw["input_line_no"] = line_no
                raw["line_no"] = original_line_no_by_input_line_no[line_no]
                fout.write(json.dumps(raw, ensure_ascii=False) + "\n")

    sorted_suspicious_path = output_dir / "suspicious_embedding_noise_sorted.jsonl"
    with open(sorted_suspicious_path, "w", encoding="utf-8") as fout:
        for score_item in suspicious_scores:
            input_line_no = score_item["input_line_no"]
            if input_line_no not in suspicious_input_line_nos:
                continue
            fout.write(json.dumps(score_item, ensure_ascii=False) + "\n")

    summary = {
        "input_path": args.input_path,
        "model_path": args.model_path,
        "scored_path": str(scored_path),
        "errors_path": str(errors_path),
        "suspicious_path": str(suspicious_path),
        "suspicious_sorted_score_path": str(sorted_suspicious_path),
        "total": total,
        "scored": len(all_scores),
        "errors": errors,
        "suspicious": len(suspicious_scores),
        "min_score": sorted_scores[0]["embedding_similarity"] if sorted_scores else None,
        "max_score": sorted_scores[-1]["embedding_similarity"] if sorted_scores else None,
        "threshold": args.threshold,
        "top_ratio": args.top_ratio if args.threshold is None and args.top_k is None else None,
        "top_k": args.top_k,
        "max_length": max_length,
        "no_truncate": args.no_truncate,
        "device": device,
    }
    summary_path = output_dir / "embedding_noise_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
