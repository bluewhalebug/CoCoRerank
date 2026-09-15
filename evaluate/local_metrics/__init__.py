"""Self-contained BLEU, ROUGE-L, and METEOR evaluation used by the paper."""

import json
from collections import Counter, OrderedDict

from .bleu import corpus_bleu
from .meteor import Meteor
from .rouge import Rouge


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n


def calc_metrics(hypotheses, references, sources=None, filename=None, mode="train"):
    """Compute the BLEU, ROUGE-L, METEOR, and token-overlap metrics.

    ``hypotheses`` and ``references`` map sample identifiers to lists of text.
    Each hypothesis list must contain exactly one prediction. METEOR is computed
    only in ``test`` mode, matching the original experimental helper.
    """
    if sorted(references) != sorted(hypotheses):
        raise ValueError("hypotheses and references must have identical keys")

    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)
    rouge_l, ind_rouge = Rouge().compute_score(references, hypotheses)

    meteor = 0.0
    if mode == "test":
        meteor_calculator = Meteor()
        try:
            meteor, _ = meteor_calculator.compute_score(references, hypotheses)
        finally:
            meteor_calculator.close()

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    output_file = open(filename, "w", encoding="utf-8") if filename else None
    try:
        for key in references:
            item_precision, item_recall, item_f1 = calc_accs(
                hypotheses[key][0], references[key]
            )
            precision.update(item_precision)
            recall.update(item_recall)
            f1.update(item_f1)
            if output_file:
                log_obj = OrderedDict(
                    id=key,
                    predictions=hypotheses[key],
                    references=references[key],
                    bleu=ind_bleu[key],
                    rouge_l=ind_rouge[key],
                )
                if sources is not None:
                    log_obj["code"] = sources[key]
                output_file.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
    finally:
        if output_file:
            output_file.close()

    return {
        "bleu": bleu * 100,
        "rouge_l": rouge_l * 100,
        "meteor": meteor * 100,
        "precision": precision.avg * 100,
        "recall": recall.avg * 100,
        "f1": f1.avg * 100,
    }


def calc_accs(prediction, ground_truths):
    precision = recall = f1 = 0.0
    for ground_truth in ground_truths:
        item_precision, item_recall, item_f1 = calc_acc(
            prediction, ground_truth
        )
        if item_f1 > f1:
            precision, recall, f1 = item_precision, item_recall, item_f1
    return precision, recall, f1


def calc_acc(prediction, ground_truth):
    """Compute token-level precision, recall, and F1 after normalization."""
    if not ground_truth:
        return (1.0, 1.0, 1.0) if not prediction else (0.0, 0.0, 0.0)

    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if not num_same:
        return 0.0, 0.0, 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def normalize_string(text):
    return " ".join(text.lower().split())
