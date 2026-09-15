import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cluster_eval_utils import extract_subject, read_jsonl  # noqa: E402


DEFAULT_KS = [1, 2, 3, 5]


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


def load_evaluate_ranked_rows():
    from cluster_rerank_eval import evaluate_ranked_rows

    return evaluate_ranked_rows


def parse_ranked_paths(values: Optional[Sequence[str]]) -> Dict[str, str]:
    paths = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--ranked-path entries must use name=/path/to/ranked.jsonl")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("--ranked-path entries must use non-empty name and path")
        paths[name] = path
    return paths


def resolve_ranked_paths(args) -> Dict[str, str]:
    explicit_paths = parse_ranked_paths(args.ranked_path)
    if explicit_paths:
        return explicit_paths

    if not args.ranked_dir:
        raise ValueError(
            "--ranked-dir is required when --ranked-path is not provided"
        )
    rankers = args.rankers or [
        "easy_model",
        "hard_model",
        "mbr",
        "vote",
        "Majority_Voting",
    ]
    return {
        ranker_name: str(Path(args.ranked_dir) / f"{ranker_name}_cluster_ranked.jsonl")
        for ranker_name in rankers
    }


def load_ranked_rows(path: str) -> List[dict]:
    return list(read_jsonl(path))


def load_raw_candidate_firstk_rows(
    cand_path: str,
    test_path: str,
    top_k: int,
    ranker_name: str,
) -> List[dict]:
    rows = []
    sentinel = object()
    streams = zip_longest(read_jsonl(cand_path), read_jsonl(test_path), fillvalue=sentinel)

    for row_id, (cand_obj, test_obj) in enumerate(streams):
        if sentinel in (cand_obj, test_obj):
            raise ValueError("cand-path and test-path have different lengths")

        index = cand_obj.get("index", row_id)
        gt_subject = test_obj.get("subject")
        candidates = cand_obj.get("commit_message")
        if not isinstance(candidates, list) or not candidates:
            rows.append(
                {
                    "index": index,
                    "ranker": ranker_name,
                    "error": "missing_or_empty_commit_message",
                    "ground_truth_subject": gt_subject,
                }
            )
            continue

        subjects = [
            subject
            for subject in (extract_subject(candidate) for candidate in candidates[:top_k])
            if isinstance(subject, str) and subject.strip()
        ]
        if not subjects:
            rows.append(
                {
                    "index": index,
                    "ranker": ranker_name,
                    "error": "no_valid_subjects_in_first_candidates",
                    "ground_truth_subject": gt_subject,
                }
            )
            continue

        rows.append(
            {
                "index": index,
                "ranker": ranker_name,
                "top1_index": 0,
                "top1_subject": subjects[0],
                "global_order_indices": list(range(len(subjects))),
                "global_order_subjects": subjects,
                "pass_order_indices": list(range(len(subjects))),
                "pass_order_subjects": subjects,
                "ground_truth_subject": gt_subject,
                "raw_cand_path": cand_path,
                "raw_first_k": top_k,
            }
        )

    return rows


def extract_type(message: object) -> Optional[str]:
    if not isinstance(message, str):
        return None
    message = message.strip()
    if not message:
        return None
    cut_positions = [pos for pos in (message.find("("), message.find(":")) if pos >= 0]
    if cut_positions:
        message = message[: min(cut_positions)]
    else:
        message = message.split(None, 1)[0]
    message = message.strip().lower()
    return message or None


def normalize_type(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value or None


def order_types_by_cluster_size(messages: Sequence[object]) -> List[dict]:
    clusters = {}
    for idx, message in enumerate(messages):
        commit_type = extract_type(message)
        if not commit_type:
            continue
        if commit_type not in clusters:
            clusters[commit_type] = {
                "type": commit_type,
                "count": 0,
                "first_index": idx,
                "candidate_indices": [],
            }
        clusters[commit_type]["count"] += 1
        clusters[commit_type]["candidate_indices"].append(idx)

    return sorted(
        clusters.values(),
        key=lambda item: (-item["count"], item["first_index"], item["type"]),
    )


def evaluate_raw_type_pass(
    cand_path: str,
    test_path: str,
    ks: Sequence[int],
    ranker_name: str,
) -> dict:
    stats = {k: {"hit": 0, "total": 0, "pass": 0.0} for k in ks}
    samples = 0
    skipped = 0
    sentinel = object()
    streams = zip_longest(read_jsonl(cand_path), read_jsonl(test_path), fillvalue=sentinel)

    for row_id, (cand_obj, test_obj) in enumerate(streams):
        if sentinel in (cand_obj, test_obj):
            raise ValueError("cand-path and test-path have different lengths")

        gt_type = normalize_type(test_obj.get("type"))
        messages = cand_obj.get("commit_message")
        if not gt_type or not isinstance(messages, list) or not messages:
            skipped += 1
            continue

        ordered_clusters = order_types_by_cluster_size(messages)
        ordered_types = [cluster["type"] for cluster in ordered_clusters]
        if not ordered_types:
            skipped += 1
            continue

        samples += 1
        for k in ks:
            stats[k]["total"] += 1
            if gt_type in ordered_types[:k]:
                stats[k]["hit"] += 1

    for k, values in stats.items():
        values["pass"] = values["hit"] / values["total"] if values["total"] else 0.0

    return {
        "ranker": ranker_name,
        "cand_path": cand_path,
        "test_path": test_path,
        "pass_at_k": stats,
        "samples": samples,
        "skipped": skipped,
        "match_rule": "exact_type_match",
        "cluster_rule": "candidate commit messages grouped by extracted type; clusters sorted by size desc",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved cluster-aware ranked jsonl files. This script never runs "
            "rerank models, so it is the right entry point for changing hit threshold, "
            "BLEU order, or pass@k values."
        )
    )
    parser.add_argument(
        "--ranked-dir",
        help="Directory containing {ranker}_cluster_ranked.jsonl files.",
    )
    parser.add_argument(
        "--rankers",
        nargs="+",
        default=None,
        help="Ranker names to load from --ranked-dir when --ranked-path is not used.",
    )
    parser.add_argument(
        "--ranked-path",
        action="append",
        default=[],
        help="Explicit ranked file path as name=/path/to/ranked.jsonl. Can repeat.",
    )
    parser.add_argument(
        "--skip-ranked",
        action="store_true",
        help="Do not load saved ranked files; useful when only evaluating raw cand pass@k.",
    )
    parser.add_argument(
        "--summary-output",
        required=True,
        help="Path for the JSON summary.",
    )
    parser.add_argument(
        "--include-raw-cand-pass",
        action="store_true",
        help=(
            "Also evaluate pass@k directly from the first candidates in --raw-cand-path, "
            "without clustering."
        ),
    )
    parser.add_argument("--raw-cand-path")
    parser.add_argument("--test-path")
    parser.add_argument("--raw-cand-name", default="raw_cand_prefix")
    parser.add_argument(
        "--include-raw-type-pass",
        action="store_true",
        help=(
            "Also evaluate type pass@k from --raw-cand-path. Candidate commit "
            "messages are grouped by extracted type, clusters are sorted by size, "
            "and hits are exact matches against testset type."
        ),
    )
    parser.add_argument("--raw-type-name", default="raw_type_cluster_size")
    parser.add_argument(
        "--raw-cand-top-k",
        type=int,
        default=None,
        help=(
            "Number of leading candidates to use from raw cand jsonl. "
            "Defaults to max(--ks)."
        ),
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
            "w/o Vertical Consistency ablation. "
            "--include-raw-cand-pass remains in original candidate order."
        ),
    )
    parser.add_argument(
        "--bleu-smooth",
        type=float,
        default=0.0,
        help="Optional additive smoothing for the custom pass@k BLEU scorer.",
    )
    parser.add_argument("--metric-mode", choices=["dev", "test"], default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.include_raw_cand_pass or args.include_raw_type_pass) and (
        not args.raw_cand_path or not args.test_path
    ):
        raise ValueError(
            "--raw-cand-path and --test-path are required for raw candidate evaluation"
        )
    raw_cand_top_k = args.raw_cand_top_k if args.raw_cand_top_k is not None else max(args.ks)
    if raw_cand_top_k <= 0:
        raise ValueError("--raw-cand-top-k must be positive")
    ranked_paths = {} if args.skip_ranked else resolve_ranked_paths(args)
    summaries = {}
    needs_subject_eval = bool(ranked_paths) or args.include_raw_cand_pass
    evaluate_ranked_rows = load_evaluate_ranked_rows() if needs_subject_eval else None

    for ranker_name, ranked_path in ranked_paths.items():
        ranked_rows = load_ranked_rows(ranked_path)
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
        summary["ranked_path"] = ranked_path
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

    if args.include_raw_cand_pass:
        raw_rows = load_raw_candidate_firstk_rows(
            cand_path=args.raw_cand_path,
            test_path=args.test_path,
            top_k=raw_cand_top_k,
            ranker_name=args.raw_cand_name,
        )
        raw_summary = evaluate_ranked_rows(
            ranked_rows=raw_rows,
            ks=args.ks,
            bleu_order=args.bleu_order,
            hit_threshold=args.hit_threshold,
            bleu_smooth=args.bleu_smooth,
            metric_mode=args.metric_mode,
            cluster_aware_eval=False,
        )
        raw_summary["ranker"] = args.raw_cand_name
        raw_summary["raw_cand_path"] = args.raw_cand_path
        raw_summary["test_path"] = args.test_path
        raw_summary["raw_cand_top_k"] = raw_cand_top_k
        summaries[args.raw_cand_name] = raw_summary

        print(f"\n=== {args.raw_cand_name} ===")
        for k in args.ks:
            values = raw_summary["pass_at_k"][k]
            print(
                f"pass@{k}: {values['pass']:.4f} "
                f"({values['hit']}/{values['total']})"
            )
        for key, value in raw_summary["top1_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    if args.include_raw_type_pass:
        type_summary = evaluate_raw_type_pass(
            cand_path=args.raw_cand_path,
            test_path=args.test_path,
            ks=args.ks,
            ranker_name=args.raw_type_name,
        )
        summaries[args.raw_type_name] = type_summary

        print(f"\n=== {args.raw_type_name} ===")
        for k in args.ks:
            values = type_summary["pass_at_k"][k]
            print(
                f"pass@{k}: {values['pass']:.4f} "
                f"({values['hit']}/{values['total']})"
            )
        print(f"samples: {type_summary['samples']}, skipped: {type_summary['skipped']}")

    full_summary = {
        "ranked_paths": ranked_paths,
        "include_raw_cand_pass": args.include_raw_cand_pass,
        "include_raw_type_pass": args.include_raw_type_pass,
        "raw_cand_path": (
            args.raw_cand_path
            if args.include_raw_cand_pass or args.include_raw_type_pass
            else None
        ),
        "test_path": (
            args.test_path
            if args.include_raw_cand_pass or args.include_raw_type_pass
            else None
        ),
        "raw_cand_top_k": raw_cand_top_k if args.include_raw_cand_pass else None,
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
