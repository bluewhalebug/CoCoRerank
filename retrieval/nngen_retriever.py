import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:  # pragma: no cover - handled at runtime for mixed mode.
    SmoothingFunction = None
    sentence_bleu = None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def build_message(item: Dict[str, Any], mode: str = "subject") -> str:
    if mode == "subject":
        return normalize_text(item.get("subject"))
    if mode == "msg":
        return normalize_text(item.get("msg"))
    if mode == "type_scope_subject":
        commit_type = normalize_text(item.get("type"))
        scope = normalize_text(item.get("scope"))
        subject = normalize_text(item.get("subject"))
        if commit_type and scope:
            return f"{commit_type}({scope}):{subject}"
        if commit_type:
            return f"{commit_type}:{subject}"
        return subject
    raise ValueError(f"Unsupported message mode: {mode}")


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return len(text.split()) if text.strip() else 0
    return len(tokenizer.encode(text, add_special_tokens=False))


class NNGenRetriever:
    """NNGen-style nearest-neighbor retriever for building ICL examples.

    The original NNGen implementation builds a full dense test-by-train cosine
    matrix. This class keeps the same CountVectorizer cosine scoring idea but
    computes one query at a time, which is practical for larger JSONL datasets.
    """

    def __init__(
        self,
        data: Sequence[Dict[str, Any]],
        *,
        commit_field: str = "commit",
        message_mode: str = "subject",
        nn_type: str = "cos",
        top_k: int = 5,
        search_k: Optional[int] = None,
        tokenizer=None,
    ) -> None:
        if nn_type not in {"cos", "mixed"}:
            raise ValueError("nn_type must be 'cos' or 'mixed'")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.data = list(data)
        self.commit_field = commit_field
        self.message_mode = message_mode
        self.nn_type = nn_type
        self.top_k = top_k
        self.search_k = max(search_k or top_k, top_k)
        self.tokenizer = tokenizer

        self.train_diffs = [normalize_text(item.get(commit_field)) for item in self.data]
        self.train_msgs = [build_message(item, mode=message_mode) for item in self.data]
        self.vectorizer = CountVectorizer()
        train_matrix = self.vectorizer.fit_transform(self.train_diffs)
        self.train_matrix = normalize(train_matrix, norm="l2", copy=False)

        if self.nn_type == "mixed" and sentence_bleu is None:
            raise ImportError("nltk is required for nn_type='mixed'")
        self._bleu_smoother = (
            SmoothingFunction().method1
            if self.nn_type == "mixed" and SmoothingFunction is not None
            else None
        )

    def retrieve(
        self,
        query_item: Dict[str, Any],
        *,
        max_tokens: Optional[int] = None,
        exclude_sha: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        query_diff = normalize_text(query_item.get(self.commit_field))
        if not query_diff or not self.data:
            return [], [], []

        query_matrix = self.vectorizer.transform([query_diff])
        query_matrix = normalize(query_matrix, norm="l2", copy=False)
        scores = (query_matrix @ self.train_matrix.T).toarray()[0]

        base_candidate_count = self.search_k if self.nn_type == "mixed" else self.top_k
        if exclude_sha:
            base_candidate_count += 1
        candidate_count = min(len(scores), base_candidate_count)
        if candidate_count <= 0:
            return [], [], []

        candidate_idx = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

        if exclude_sha:
            candidate_idx = np.array(
                [idx for idx in candidate_idx if self.data[int(idx)].get("sha") != exclude_sha],
                dtype=int,
            )

        if self.nn_type == "mixed":
            ranked = self._rerank_with_bleu(candidate_idx, query_diff, scores)
        else:
            ranked = [(int(idx), float(scores[int(idx)]), None) for idx in candidate_idx]

        icl_commits: List[str] = []
        icl_msgs: List[str] = []
        metadata: List[Dict[str, Any]] = []
        used_tokens = 0

        for idx, cosine_score, bleu_score in ranked:
            if len(icl_commits) >= self.top_k:
                break

            src = self.train_diffs[idx]
            dst = self.train_msgs[idx]
            pair_tokens = count_tokens(src, self.tokenizer) + count_tokens(dst, self.tokenizer)
            if max_tokens is not None and used_tokens + pair_tokens > max_tokens:
                continue

            item = self.data[idx]
            icl_commits.append(src)
            icl_msgs.append(dst)
            used_tokens += pair_tokens
            metadata.append(
                {
                    "sha": item.get("sha"),
                    "index": idx,
                    "cosine": cosine_score,
                    "bleu": bleu_score,
                    "tokens": pair_tokens,
                }
            )

        return icl_commits, icl_msgs, metadata

    def _rerank_with_bleu(
        self,
        candidate_idx: Sequence[int],
        query_diff: str,
        cosine_scores: np.ndarray,
    ) -> List[Tuple[int, float, float]]:
        query_tokens = query_diff.split()
        ranked: List[Tuple[int, float, float]] = []
        for idx in candidate_idx:
            idx = int(idx)
            candidate_tokens = self.train_diffs[idx].split()
            if not candidate_tokens or not query_tokens:
                bleu_score = 0.0
            else:
                bleu_score = float(
                    sentence_bleu(
                        [candidate_tokens],
                        query_tokens,
                        smoothing_function=self._bleu_smoother,
                    )
                )
                if math.isnan(bleu_score):
                    bleu_score = 0.0
            ranked.append((idx, float(cosine_scores[idx]), bleu_score))

        ranked.sort(key=lambda item: (item[2], item[1]), reverse=True)
        return ranked
