import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: object) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = " ".join(text.strip().lower().split())
    return text or None


def normalize_for_match(text: object) -> Optional[str]:
    text = normalize(text)
    if not text:
        return None
    text = text.strip(" \t\r\n\"'`“”‘’")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.strip(" \t\r\n\"'`“”‘’")
    return text or None


def weak_match_normalized(pred: Optional[str], target: Optional[str]) -> bool:
    if not pred or not target:
        return False
    if pred == target:
        return True

    pred_no_ticks = pred.replace("`", "")
    target_no_ticks = target.replace("`", "")
    if pred_no_ticks == target_no_ticks:
        return True

    pred_light = pred_no_ticks.strip(".,;:!?\"'")
    target_light = target_no_ticks.strip(".,;:!?\"'")
    if pred_light == target_light:
        return True

    longer = max(len(pred_light), len(target_light))
    shorter = min(len(pred_light), len(target_light))
    if longer >= 20 and longer - shorter <= 3:
        return SequenceMatcher(None, pred_light, target_light).ratio() >= 0.97

    return False


def weak_match(pred: object, target: object) -> bool:
    return weak_match_normalized(normalize_for_match(pred), normalize_for_match(target))


def extract_subject(message: object) -> Optional[str]:
    if not isinstance(message, str):
        return None
    message = message.strip()
    if ":" in message:
        return message.split(":", 1)[1].strip() or None
    if ")" in message:
        return message.split(")", 1)[1].strip() or None
    return message or None


def clean_metric_text(text: str) -> str:
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text.replace("|||", " ").replace("\n", " ").replace("\r", " ").strip()


def hard_match_key(subject: object) -> Optional[str]:
    return normalize(subject)


@dataclass
class WeakCluster:
    cluster_id: int
    candidate_indices: List[int]
    size: int
    first_index: int


def _find(parent: List[int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def build_weak_clusters(subjects: Sequence[Optional[str]]) -> List[WeakCluster]:
    """Cluster candidates with the same weak-match rule used by baseline/ranker/vote.py."""
    entries = []
    hard_counts = compute_hard_match_counts(subjects)

    for idx, subject in enumerate(subjects):
        key = hard_match_key(subject)
        match_key = normalize_for_match(subject)
        if not key or not match_key:
            continue
        entries.append(
            {
                "candidate_index": idx,
                "hard_key": key,
                "match_key": match_key,
                "hard_count": hard_counts[idx],
            }
        )

    if not entries:
        return []

    parent = list(range(len(entries)))

    def union(left: int, right: int) -> None:
        left_root = _find(parent, left)
        right_root = _find(parent, right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if weak_match_normalized(entries[i]["match_key"], entries[j]["match_key"]):
                union(i, j)

    grouped = defaultdict(list)
    for entry_idx, entry in enumerate(entries):
        grouped[_find(parent, entry_idx)].append(entry["candidate_index"])

    clusters = []
    for cluster_id, candidate_indices in enumerate(grouped.values()):
        candidate_indices = sorted(candidate_indices)
        clusters.append(
            WeakCluster(
                cluster_id=cluster_id,
                candidate_indices=candidate_indices,
                size=len(candidate_indices),
                first_index=min(candidate_indices),
            )
        )

    clusters.sort(key=lambda cluster: cluster.first_index)
    for cluster_id, cluster in enumerate(clusters):
        cluster.cluster_id = cluster_id
    return clusters


def compute_hard_match_counts(subjects: Sequence[Optional[str]]) -> List[int]:
    keys = [hard_match_key(subject) for subject in subjects]
    counts = Counter(key for key in keys if key)
    return [counts.get(key, 0) if key else 0 for key in keys]


def stable_softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    denom = sum(exp_scores)
    if denom <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [score / denom for score in exp_scores]


def order_for_pass_at_k(
    clusters: Sequence[WeakCluster],
    candidate_order_value: Sequence[float],
    cluster_order_value: Dict[int, float],
) -> List[int]:
    """Round-robin candidates across sorted clusters as requested for pass@k."""
    sorted_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -cluster_order_value.get(cluster.cluster_id, 0.0),
            -cluster.size,
            cluster.first_index,
        ),
    )

    ordered_within_cluster = []
    for cluster in sorted_clusters:
        ordered_indices = sorted(
            cluster.candidate_indices,
            key=lambda idx: (-candidate_order_value[idx], idx),
        )
        ordered_within_cluster.append(ordered_indices)

    pass_order = []
    max_depth = max((len(indices) for indices in ordered_within_cluster), default=0)
    for depth in range(max_depth):
        for indices in ordered_within_cluster:
            if depth < len(indices):
                pass_order.append(indices[depth])
    return pass_order


def tokenize_for_bleu(text: object) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(
    prediction: object,
    reference: object,
    max_order: int = 1,
    smooth: float = 0.0,
) -> float:
    """Return sentence BLEU in percentage form, independent from calc_metrics."""
    if max_order < 1 or max_order > 4:
        raise ValueError("max_order must be in [1, 4]")

    pred_tokens = tokenize_for_bleu(prediction)
    ref_tokens = tokenize_for_bleu(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        total = sum(pred_counts.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_counts[ngram]) for ngram, count in pred_counts.items())
        if smooth > 0:
            precisions.append((overlap + smooth) / (total + smooth))
        else:
            precisions.append(overlap / total)

    if any(precision <= 0 for precision in precisions):
        return 0.0

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    geo_mean = math.exp(sum(math.log(precision) for precision in precisions) / max_order)
    return 100.0 * brevity_penalty * geo_mean


def is_bleu_hit(
    prediction: object,
    reference: object,
    max_order: int,
    threshold: float,
    inclusive: bool = False,
    smooth: float = 0.0,
) -> bool:
    score = sentence_bleu(prediction, reference, max_order=max_order, smooth=smooth)
    if inclusive:
        return score >= threshold
    return score > threshold
