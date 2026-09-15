import argparse
import importlib.util
import json
import os
import sys
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cluster_eval_utils import (  # noqa: E402
    WeakCluster,
    build_weak_clusters,
    clean_metric_text,
    compute_hard_match_counts,
    extract_subject,
    is_bleu_hit,
    order_for_pass_at_k,
    read_jsonl,
    stable_softmax,
)
from local_metrics import calc_metrics  # noqa: E402


DEFAULT_KS = [1, 2, 3, 5]

MODEL_FREE_RANKERS = {"mbr", "vote", "Majority_Voting"}


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list cannot be empty")
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer: {item}") from exc
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all k values must be positive")
    return parsed


def parse_model_overrides(values: Optional[Sequence[str]]) -> Dict[str, str]:
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--model-path entries must use name=/path/to/model.pt")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("--model-path entries must use non-empty name and path")
        overrides[name] = path
    return overrides


def normalize_field(value: object, default: str = "<none>") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def load_rerank_module(rerank_module_path: str):
    spec = importlib.util.spec_from_file_location(
        "existing_consist_rerank", rerank_module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rerank module from {rerank_module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_input_rows(
    cand_path: str,
    test_path: str,
    vote_meta_path: Optional[str],
    strict_index: bool = False,
) -> Iterable[dict]:
    sentinel = object()
    if vote_meta_path:
        streams = zip_longest(
            read_jsonl(cand_path),
            read_jsonl(test_path),
            read_jsonl(vote_meta_path),
            fillvalue=sentinel,
        )
    else:
        streams = zip_longest(
            read_jsonl(cand_path),
            read_jsonl(test_path),
            fillvalue=sentinel,
        )

    for row_id, packed in enumerate(streams):
        if vote_meta_path:
            cand_obj, test_obj, vote_obj = packed
        else:
            cand_obj, test_obj = packed
            vote_obj = {}

        if sentinel in packed:
            raise ValueError("Input jsonl files have different lengths")

        index = cand_obj.get("index", row_id)
        test_index = test_obj.get("index")
        vote_index = vote_obj.get("index") if isinstance(vote_obj, dict) else None

        if strict_index and test_index is not None and test_index != index:
            raise ValueError(
                f"Index mismatch at row {row_id}: cand={index}, test={test_index}"
            )
        if strict_index and vote_index is not None and vote_index != index:
            raise ValueError(
                f"Index mismatch at row {row_id}: cand={index}, vote={vote_index}"
            )

        if "error" in cand_obj:
            yield {
                "index": index,
                "error": cand_obj["error"],
                "ground_truth_subject": test_obj.get("subject"),
            }
            continue

        candidates = cand_obj.get("commit_message")
        if not isinstance(candidates, list) or not candidates:
            yield {
                "index": index,
                "error": "missing_or_empty_commit_message",
                "ground_truth_subject": test_obj.get("subject"),
            }
            continue

        commit = normalize_field(test_obj.get("commit"), "")
        commit_type = normalize_field(vote_obj.get("most_common_type"))
        scope = normalize_field(vote_obj.get("most_common_scope"))

        yield {
            "index": index,
            "candidates": candidates,
            "subjects": [extract_subject(candidate) for candidate in candidates],
            "ground_truth_subject": test_obj.get("subject"),
            "commit": commit,
            "meta": f"type: {commit_type}\nscope: {scope}",
        }


def cluster_summary(
    clusters: Sequence[WeakCluster],
    subjects: Sequence[Optional[str]],
    candidate_order_value: Sequence[float],
    cluster_order_value: Dict[int, float],
) -> List[dict]:
    summaries = []
    for cluster in sorted(
        clusters,
        key=lambda item: (
            -cluster_order_value.get(item.cluster_id, 0.0),
            -item.size,
            item.first_index,
        ),
    ):
        ordered_indices = sorted(
            cluster.candidate_indices,
            key=lambda idx: (-candidate_order_value[idx], idx),
        )
        summaries.append(
            {
                "cluster_id": cluster.cluster_id,
                "cluster_order_value": cluster_order_value.get(cluster.cluster_id, 0.0),
                "size": cluster.size,
                "candidate_indices": ordered_indices,
                "subjects": [subjects[idx] for idx in ordered_indices],
            }
        )
    return summaries


def build_ranked_result(
    row: dict,
    ranker_name: str,
    scores: Sequence[float],
    probs: Optional[Sequence[float]],
    candidate_order_value: Sequence[float],
    cluster_order_value: Dict[int, float],
    global_order_indices: Sequence[int],
) -> dict:
    subjects = row["subjects"]
    clusters = build_weak_clusters(subjects)
    pass_order_indices = order_for_pass_at_k(
        clusters=clusters,
        candidate_order_value=candidate_order_value,
        cluster_order_value=cluster_order_value,
    )

    top1_index = global_order_indices[0] if global_order_indices else None
    top1_subject = subjects[top1_index] if top1_index is not None else None

    return {
        "index": row["index"],
        "ranker": ranker_name,
        "top1_index": top1_index,
        "top1_subject": top1_subject,
        "global_order_indices": list(global_order_indices),
        "global_order_subjects": [subjects[idx] for idx in global_order_indices],
        "pass_order_indices": pass_order_indices,
        "pass_order_subjects": [subjects[idx] for idx in pass_order_indices],
        "candidate_scores": list(scores),
        "candidate_probs": list(probs) if probs is not None else None,
        "ground_truth_subject": row.get("ground_truth_subject"),
        "clusters": cluster_summary(
            clusters=clusters,
            subjects=subjects,
            candidate_order_value=candidate_order_value,
            cluster_order_value=cluster_order_value,
        ),
    }


def build_model_ranked_result(
    row: dict,
    ranker_name: str,
    scores: Sequence[float],
    probs: Sequence[float],
) -> dict:
    clusters = build_weak_clusters(row["subjects"])
    cluster_probs = {
        cluster.cluster_id: sum(probs[idx] for idx in cluster.candidate_indices)
        for cluster in clusters
    }
    global_order = sorted(range(len(row["subjects"])), key=lambda idx: (-scores[idx], idx))
    return build_ranked_result(
        row=row,
        ranker_name=ranker_name,
        scores=scores,
        probs=probs,
        candidate_order_value=scores,
        cluster_order_value=cluster_probs,
        global_order_indices=global_order,
    )


def tokenize_for_mbr(text: str) -> List[str]:
    import re

    return re.findall(r"\b\w+\b", text.lower())


def mbr_candidate_scores(subjects: Sequence[Optional[str]]) -> List[float]:
    from rank_bm25 import BM25Okapi

    valid_subjects = []
    seen = set()
    for subject in subjects:
        if not isinstance(subject, str) or not subject.strip():
            continue
        if subject in seen:
            continue
        seen.add(subject)
        valid_subjects.append(subject)

    if not valid_subjects:
        return [0.0 for _ in subjects]
    if len(valid_subjects) == 1:
        only = valid_subjects[0]
        return [1.0 if subject == only else 0.0 for subject in subjects]

    tokenized_corpus = [tokenize_for_mbr(subject) for subject in valid_subjects]
    bm25 = BM25Okapi(tokenized_corpus)
    unique_scores = {}
    for subject, query in zip(valid_subjects, tokenized_corpus):
        unique_scores[subject] = float(bm25.get_scores(query).mean())

    return [
        unique_scores.get(subject, 0.0) if isinstance(subject, str) else 0.0
        for subject in subjects
    ]


def rank_mbr_rows(rows: Sequence[dict], ranker_name: str) -> List[dict]:
    ranked_rows = []
    for row in tqdm(rows, desc=ranker_name):
        if "error" in row:
            ranked_rows.append(
                {
                    "index": row["index"],
                    "ranker": ranker_name,
                    "error": row["error"],
                    "ground_truth_subject": row.get("ground_truth_subject"),
                }
            )
            continue

        scores = mbr_candidate_scores(row["subjects"])
        probs = stable_softmax(scores)
        ranked_rows.append(build_model_ranked_result(row, ranker_name, scores, probs))
    return ranked_rows


def rank_vote_rows(rows: Sequence[dict], ranker_name: str) -> List[dict]:
    ranked_rows = []
    for row in tqdm(rows, desc=ranker_name):
        if "error" in row:
            ranked_rows.append(
                {
                    "index": row["index"],
                    "ranker": ranker_name,
                    "error": row["error"],
                    "ground_truth_subject": row.get("ground_truth_subject"),
                }
            )
            continue

        subjects = row["subjects"]
        clusters = build_weak_clusters(subjects)
        hard_counts = compute_hard_match_counts(subjects)
        cluster_sizes = {cluster.cluster_id: float(cluster.size) for cluster in clusters}
        global_order = sorted(range(len(subjects)), key=lambda idx: (-hard_counts[idx], idx))
        ranked_rows.append(
            build_ranked_result(
                row=row,
                ranker_name=ranker_name,
                scores=hard_counts,
                probs=None,
                candidate_order_value=hard_counts,
                cluster_order_value=cluster_sizes,
                global_order_indices=global_order,
            )
        )
    return ranked_rows


def rank_majority_voting_rows(
    rows: Sequence[dict], ranker_name: str
) -> List[dict]:
    ranked_rows = []
    for row in tqdm(rows, desc=ranker_name):
        if "error" in row:
            ranked_rows.append(
                {
                    "index": row["index"],
                    "ranker": ranker_name,
                    "error": row["error"],
                    "ground_truth_subject": row.get("ground_truth_subject"),
                }
            )
            continue

        subjects = row["subjects"]
        scores = compute_hard_match_counts(subjects)
        global_order = sorted(range(len(subjects)), key=lambda idx: (-scores[idx], idx))
        ranked_rows.append(
            {
                "index": row["index"],
                "ranker": ranker_name,
                "top1_index": global_order[0] if global_order else None,
                "top1_subject": subjects[global_order[0]] if global_order else None,
                "global_order_indices": global_order,
                "global_order_subjects": [subjects[idx] for idx in global_order],
                "pass_order_indices": global_order,
                "pass_order_subjects": [subjects[idx] for idx in global_order],
                "candidate_scores": scores,
                "candidate_probs": None,
                "ground_truth_subject": row.get("ground_truth_subject"),
                "clusters": [],
            }
        )
    return ranked_rows


def load_model_components(args, model_path: str):
    rerank_module = load_rerank_module(args.rerank_module_path)
    checkpoint, state_dict = rerank_module.load_checkpoint(model_path)
    resolve_args = SimpleNamespace(
        model_name=args.model_name,
        commit_max_len=args.commit_max_len,
        meta_max_len=args.meta_max_len,
        subject_max_len=args.subject_max_len,
        pooling=args.pooling,
        score_temperature=args.score_temperature,
    )
    (
        model_name,
        commit_max_len,
        meta_max_len,
        subject_max_len,
        pooling,
        score_temperature,
    ) = rerank_module.resolve_config(resolve_args, checkpoint)
    num_types = rerank_module.infer_num_types(checkpoint, state_dict)
    device = rerank_module.get_device(args.device)
    amp_dtype = rerank_module.get_amp_dtype(device, args.amp_dtype)

    if device.type == "cuda":
        rerank_module.torch.backends.cuda.matmul.allow_tf32 = True
        rerank_module.torch.set_float32_matmul_precision("high")

    tokenizer = rerank_module.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = rerank_module.StructuredBiEncoder(model_name, num_types, pooling)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()

    cfg = {
        "commit_max_len": commit_max_len,
        "meta_max_len": meta_max_len,
        "subject_max_len": subject_max_len,
        "subject_batch_size": args.subject_batch_size,
        "score_temperature": score_temperature,
    }
    return rerank_module, model, tokenizer, cfg, device, amp_dtype


def score_model_batch(
    rerank_module,
    model,
    tokenizer,
    rows: Sequence[dict],
    cfg: dict,
    device,
    amp_dtype,
) -> List[dict]:
    commits = [row["commit"] for row in rows]
    metas = [row["meta"] for row in rows]
    counts = [len(row["subjects"]) for row in rows]
    all_subjects = [subject or "" for row in rows for subject in row["subjects"]]

    commit_enc = rerank_module.tokenize_to_device(
        tokenizer, commits, cfg["commit_max_len"], device
    )
    meta_enc = rerank_module.tokenize_to_device(
        tokenizer, metas, cfg["meta_max_len"], device
    )

    with rerank_module.torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        ctx_emb, _ = model.encode_context(
            commit_enc["input_ids"],
            commit_enc["attention_mask"],
            meta_enc["input_ids"],
            meta_enc["attention_mask"],
        )

    sub_emb = rerank_module.encode_subjects(
        model,
        tokenizer,
        all_subjects,
        cfg["subject_max_len"],
        cfg["subject_batch_size"],
        device,
        amp_dtype,
    )

    if cfg["score_temperature"] <= 0:
        raise ValueError("score_temperature must be positive")

    outputs = []
    offset = 0
    same_count = len(set(counts)) == 1
    if same_count:
        cand_count = counts[0]
        score_tensor = rerank_module.torch.einsum(
            "bd,bnd->bn",
            ctx_emb,
            sub_emb.view(len(rows), cand_count, -1),
        )
        score_tensor = score_tensor / cfg["score_temperature"]
        prob_tensor = rerank_module.torch.softmax(score_tensor, dim=1)
        for scores, probs in zip(score_tensor, prob_tensor):
            outputs.append(
                {
                    "scores": scores.detach().cpu().tolist(),
                    "probs": probs.detach().cpu().tolist(),
                }
            )
        return outputs

    for row_index, count in enumerate(counts):
        scores = rerank_module.torch.matmul(
            ctx_emb[row_index], sub_emb[offset : offset + count].t()
        )
        scores = scores / cfg["score_temperature"]
        probs = rerank_module.torch.softmax(scores, dim=0)
        outputs.append(
            {
                "scores": scores.detach().cpu().tolist(),
                "probs": probs.detach().cpu().tolist(),
            }
        )
        offset += count
    return outputs


def rank_model_rows(
    rows: Sequence[dict],
    ranker_name: str,
    model_path: str,
    args,
) -> List[dict]:
    rerank_module, model, tokenizer, cfg, device, amp_dtype = load_model_components(
        args, model_path
    )
    ranked_rows = []
    pending = []

    def flush_pending() -> None:
        if not pending:
            return
        with rerank_module.torch.inference_mode():
            scored = score_model_batch(
                rerank_module,
                model,
                tokenizer,
                pending,
                cfg,
                device,
                amp_dtype,
            )
        for row, score_obj in zip(pending, scored):
            ranked_rows.append(
                build_model_ranked_result(
                    row,
                    ranker_name,
                    score_obj["scores"],
                    score_obj["probs"],
                )
            )
        pending.clear()

    for row in tqdm(rows, desc=ranker_name):
        if "error" in row:
            flush_pending()
            ranked_rows.append(
                {
                    "index": row["index"],
                    "ranker": ranker_name,
                    "error": row["error"],
                    "ground_truth_subject": row.get("ground_truth_subject"),
                }
            )
            continue
        if not row.get("commit"):
            flush_pending()
            ranked_rows.append(
                {
                    "index": row["index"],
                    "ranker": ranker_name,
                    "error": "missing_commit",
                    "ground_truth_subject": row.get("ground_truth_subject"),
                }
            )
            continue

        pending.append(row)
        if len(pending) >= args.batch_size:
            flush_pending()

    flush_pending()

    del model
    if device.type == "cuda":
        rerank_module.torch.cuda.empty_cache()
    return ranked_rows


def evaluate_ranked_rows(
    ranked_rows: Sequence[dict],
    ks: Sequence[int],
    bleu_order: int,
    hit_threshold: float,
    bleu_smooth: float,
    metric_mode: str,
    cluster_aware_eval: bool = False,
) -> dict:
    pass_stats = {k: {"hit": 0, "total": 0, "pass": 0.0} for k in ks}
    hypotheses = {}
    references = {}
    top1_skipped = 0

    for row in tqdm(ranked_rows, desc="evaluate"):
        if "error" in row:
            top1_skipped += 1
            continue

        gt_subject = row.get("ground_truth_subject")
        if not isinstance(gt_subject, str) or not gt_subject.strip():
            top1_skipped += 1
            continue

        ordered_subjects = (
            row.get("pass_order_subjects", [])
            if cluster_aware_eval
            else row.get("global_order_subjects", [])
        ) or []
        if not isinstance(ordered_subjects, list):
            ordered_subjects = []
        if not ordered_subjects and not cluster_aware_eval:
            ordered_subjects = [row.get("top1_subject")]
        top1_subject = ordered_subjects[0] if ordered_subjects else None
        if isinstance(top1_subject, str) and top1_subject.strip():
            hypotheses[row["index"]] = [clean_metric_text(top1_subject)]
            references[row["index"]] = [clean_metric_text(gt_subject)]
        else:
            top1_skipped += 1

        pass_subjects = [
            subject
            for subject in ordered_subjects
            if isinstance(subject, str) and subject.strip()
        ]
        if not pass_subjects:
            continue

        for k in ks:
            pass_stats[k]["total"] += 1
            if any(
                is_bleu_hit(
                    prediction=subject,
                    reference=gt_subject,
                    max_order=bleu_order,
                    threshold=hit_threshold,
                    inclusive=True,
                    smooth=bleu_smooth,
                )
                for subject in pass_subjects[:k]
            ):
                pass_stats[k]["hit"] += 1

    for k, values in pass_stats.items():
        values["pass"] = values["hit"] / values["total"] if values["total"] else 0.0

    if not hypotheses:
        metric_result = {}
    else:
        metric_result = calc_metrics(hypotheses, references, mode=metric_mode)

    return {
        "pass_at_k": pass_stats,
        "top1_metrics": metric_result,
        "top1_metric_samples": len(hypotheses),
        "top1_metric_skipped": top1_skipped,
        "cluster_aware_eval": cluster_aware_eval,
    }


def save_predictions(output_dir: str, ranker_name: str, rows: Sequence[dict]) -> str:
    output_path = Path(output_dir) / f"{ranker_name}_cluster_ranked.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(output_path)


def resolve_ranker_model_path(
    ranker_name: str,
    model_paths: Dict[str, str],
) -> Optional[str]:
    if ranker_name in MODEL_FREE_RANKERS:
        return None
    if ranker_name in model_paths:
        return model_paths[ranker_name]
    raise ValueError(
        f"Unknown ranker '{ranker_name}'. Known model rankers: "
        f"{sorted(model_paths)}; model-free rankers: {sorted(MODEL_FREE_RANKERS)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster-aware rerank evaluation for 100 commit-message candidates. "
            "Top1 BLEU/ROUGE/METEOR use calc_metrics; pass@k uses the global "
            "ranker order by default, or cluster-round-robin order when "
            "--cluster-aware-eval is set."
        )
    )
    parser.add_argument("--cand-path", required=True)
    parser.add_argument("--vote-meta-path")
    parser.add_argument("--test-path", required=True)
    parser.add_argument(
        "--rankers",
        nargs="+",
        default=["easy_model", "hard_model", "mbr", "vote", "Majority_Voting"],
        help=(
            "Rankers to run: hard_model (full), easy_model (w/o Hard Negatives), "
            "vote (w/o Horizontal Consistency), mbr, and Majority_Voting."
        ),
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Override or add model ranker path with name=/path/to/model.pt.",
    )
    parser.add_argument(
        "--rerank-module-path",
        default=str(CURRENT_DIR / "rerank_model.py"),
        help="Rerank-model module. Defaults to bundled evaluate/rerank_model.py.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--save-ranked",
        action="store_true",
        help="Write per-sample ranked jsonl files under --output-dir.",
    )
    parser.add_argument(
        "--summary-output",
        required=True,
        help="Path for the JSON summary.",
    )
    parser.add_argument("--ks", type=parse_int_list, default=DEFAULT_KS)
    parser.add_argument("--bleu-order", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--hit-threshold", type=float, default=70.0)
    parser.add_argument(
        "--cluster-aware-eval",
        action="store_true",
        help=(
            "Use cluster-aware pass order for pass@k and top1 metrics. "
            "This enables Vertical Consistency; omit it for the "
            "w/o Vertical Consistency ablation."
        ),
    )
    parser.add_argument(
        "--bleu-smooth",
        type=float,
        default=0.0,
        help="Optional additive smoothing for the custom pass@k BLEU scorer.",
    )
    parser.add_argument("--metric-mode", choices=["dev", "test"], default="test")
    parser.add_argument("--strict-index", action="store_true")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:1"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument(
        "--subject-batch-size",
        type=int,
        default=int(os.environ.get("SUBJECT_BATCH_SIZE", "512")),
    )
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--commit-max-len", type=int, default=None)
    parser.add_argument("--meta-max-len", type=int, default=None)
    parser.add_argument("--subject-max-len", type=int, default=None)
    parser.add_argument("--pooling", default=os.environ.get("POOLING"))
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=(
            float(os.environ["SCORE_TEMPERATURE"])
            if "SCORE_TEMPERATURE" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16", "none"],
        default=os.environ.get("AMP_DTYPE", "bf16"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_paths = parse_model_overrides(args.model_path)
    uses_model_ranker = any(
        ranker_name not in MODEL_FREE_RANKERS for ranker_name in args.rankers
    )
    if uses_model_ranker and not args.rerank_module_path:
        raise ValueError("--rerank-module-path is required for model rankers")
    if uses_model_ranker and not args.vote_meta_path:
        raise ValueError("--vote-meta-path is required for model rankers")
    if args.save_ranked and not args.output_dir:
        raise ValueError("--output-dir is required with --save-ranked")
    needs_vote_meta = uses_model_ranker
    vote_meta_path = args.vote_meta_path if needs_vote_meta else None

    rows = list(
        iter_input_rows(
            cand_path=args.cand_path,
            test_path=args.test_path,
            vote_meta_path=vote_meta_path,
            strict_index=args.strict_index,
        )
    )

    summaries = {}
    prediction_paths = {}
    for ranker_name in args.rankers:
        model_path = resolve_ranker_model_path(ranker_name, model_paths)
        if ranker_name == "mbr":
            ranked_rows = rank_mbr_rows(rows, ranker_name)
        elif ranker_name == "vote":
            ranked_rows = rank_vote_rows(rows, ranker_name)
        elif ranker_name == "Majority_Voting":
            ranked_rows = rank_majority_voting_rows(rows, ranker_name)
        else:
            ranked_rows = rank_model_rows(rows, ranker_name, model_path, args)

        if args.save_ranked:
            prediction_paths[ranker_name] = save_predictions(
                args.output_dir, ranker_name, ranked_rows
            )

        summary = evaluate_ranked_rows(
            ranked_rows=ranked_rows,
            ks=args.ks,
            bleu_order=args.bleu_order,
            hit_threshold=args.hit_threshold,
            bleu_smooth=args.bleu_smooth,
            metric_mode=args.metric_mode,
            cluster_aware_eval=args.cluster_aware_eval,
        )
        summary["ranker"] = ranker_name
        summary["model_path"] = model_path
        if ranker_name in prediction_paths:
            summary["ranked_path"] = prediction_paths[ranker_name]
        summaries[ranker_name] = summary

        print(f"\n=== {ranker_name} ===")
        for k in args.ks:
            values = summary["pass_at_k"][k]
            print(
                f"pass@{k}: {values['pass']:.4f} "
                f"({values['hit']}/{values['total']})"
            )
        for key, value in summary["top1_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    full_summary = {
        "cand_path": args.cand_path,
        "vote_meta_path": vote_meta_path,
        "test_path": args.test_path,
        "rankers": args.rankers,
        "ks": args.ks,
        "bleu_order": args.bleu_order,
        "hit_threshold": args.hit_threshold,
        "cluster_aware_eval": args.cluster_aware_eval,
        "bleu_smooth": args.bleu_smooth,
        "metric_mode": args.metric_mode,
        "summaries": summaries,
    }

    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as file_obj:
        json.dump(full_summary, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(f"\nWrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
