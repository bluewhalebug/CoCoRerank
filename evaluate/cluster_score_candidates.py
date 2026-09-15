import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from cluster_rerank_eval import (  # noqa: E402
    MODEL_FREE_RANKERS,
    iter_input_rows,
    parse_model_overrides,
    rank_mbr_rows,
    rank_model_rows,
    rank_majority_voting_rows,
    rank_vote_rows,
    resolve_ranker_model_path,
    save_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score 100 candidates and save cluster-aware ranked jsonl files. "
            "This script does not compute metrics, so threshold-only changes can be "
            "handled later by cluster_evaluate_ranked.py without rerunning models."
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
            "Rankers to score: hard_model (full), easy_model (w/o Hard Negatives), "
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
    parser.add_argument("--output-dir", required=True)
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

    written_paths = {}
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

        ranked_path = save_predictions(args.output_dir, ranker_name, ranked_rows)
        written_paths[ranker_name] = ranked_path
        print(f"{ranker_name}: wrote {ranked_path}")

    print("\nRanked files:")
    for ranker_name, ranked_path in written_paths.items():
        print(f"{ranker_name}={ranked_path}")


if __name__ == "__main__":
    main()
