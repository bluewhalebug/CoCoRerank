import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None

from nngen_retriever import NNGenRetriever


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def iter_with_progress(items: List[dict], desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=len(items), desc=desc)


def build_rows(args: argparse.Namespace) -> Iterator[dict]:
    retrieval_data = list(read_jsonl(Path(args.retrieval_path)))
    input_data = list(read_jsonl(Path(args.input_path)))
    tokenizer = load_tokenizer(args.tokenizer_path)

    retriever = NNGenRetriever(
        retrieval_data,
        commit_field=args.commit_field,
        message_mode=args.message_mode,
        nn_type=args.nn_type,
        top_k=args.top_k,
        search_k=args.search_k,
        tokenizer=tokenizer,
    )

    for idx, sample in enumerate(iter_with_progress(input_data, "Building NNGen ICL")):
        sha = sample.get("sha")
        exclude_sha = sha if args.exclude_self else None
        icl_commit, icl_msg, icl_meta = retriever.retrieve(
            sample,
            max_tokens=args.max_tokens,
            exclude_sha=exclude_sha,
        )

        sample_id = sample.get(args.id_field) if args.id_field else None
        if sample_id is None:
            sample_id = sha if sha is not None else idx

        row = {
            "id": sample_id,
            "sha": sha,
            "icl_commit": icl_commit,
            "icl_msg": icl_msg,
        }
        if args.include_meta:
            row["icl_meta"] = icl_meta
        yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ICL cache with NNGen-style nearest-neighbor retrieval."
    )
    parser.add_argument("--retrieval_path", required=True, help="JSONL retrieval/training set.")
    parser.add_argument("--input_path", required=True, help="JSONL query set to build ICL for.")
    parser.add_argument("--output_path", required=True, help="Output JSONL ICL cache.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of ICL examples per sample.")
    parser.add_argument(
        "--nn_type",
        default="cos",
        choices=["cos", "mixed"],
        help="cos: cosine only; mixed: cosine candidates reranked by BLEU."
    )
    parser.add_argument(
        "--search_k",
        type=int,
        default=None,
        help="Candidate count before BLEU rerank for --nn_type mixed. Defaults to top_k.",
    )
    parser.add_argument("--commit_field", default="commit", help="Field containing the diff text.")
    parser.add_argument(
        "--message_mode",
        default="subject",
        choices=["subject", "msg", "type_scope_subject"],
        help="How to build ICL commit messages from retrieval rows.",
    )
    parser.add_argument(
        "--id_field",
        default="sha",
        help="Field used as output id. Falls back to sha, then row index.",
    )
    parser.add_argument(
        "--exclude_self",
        action="store_true",
        help="Exclude retrieval rows with the same sha as the query sample.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional token budget for all selected ICL examples of one query.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Optional local Hugging Face tokenizer path for --max_tokens.",
    )
    parser.add_argument(
        "--include_meta",
        action="store_true",
        help="Include retrieved sha/index/scores/token counts in each output row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    write_jsonl(output_path, build_rows(args))
    print(f"[DONE] NNGen ICL cache saved to {output_path}")


if __name__ == "__main__":
    main()
