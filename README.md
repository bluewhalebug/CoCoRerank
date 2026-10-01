# CoCoRerank

This repository contains the code and dataset accompanying the paper
**“CoCoRerank: Towards Conventional Commit Message Generation by Component and
Candidate Consistency Reranking.”** 

CoCoRerank targets commit messages that follow the Conventional Commits
Specification (CCS):

type(scope): subject

The repository covers dataset normalization and semantic screening, NNGen-style
retrieval, two-stage training of the consistency scorer, cluster-aware
reranking, and evaluation. The prompts used for LLM-assisted semantic quality
assessment, candidate generation, subject-only generation, and hard-negative
generation are provided under `prompt/`. The corresponding LLM-serving code is
not included in this repository.

## Directory structure

```text
CoCoRerank/
├── README.md
├── data/
│   ├── train_set.jsonl                # Training split (52,011 samples)
│   ├── retrieval_set.jsonl            # Retrieval split (17,337 samples)
│   ├── val_set.jsonl                  # Validation split (8,670 samples)
│   └── test_set.jsonl                 # Test split (8,670 samples)
├── dataset_filter/
│   ├── clean.py                       # Structural normalization
│   └── embedding_noise_filter.py      # Embedding-based semantic screening
├── prompt/
│   ├── quality_assessment.md           # Semantic quality assessment prompt
│   ├── candidate_generation.md         # CCS candidate-generation prompt
│   ├── subject_only.md                 # Subject-only generation prompt
│   └── hard_negative.md                # Hard-negative generation prompt
├── retrieval/                          # NNGen-style retrieval
│   ├── build_icl_index.py
│   └── nngen_retriever.py
├── train/
│   ├── easytrain.py                   # Easy-stage consistency training
│   └── hardtrain.py                   # Hard-negative fine-tuning
└── evaluate/
    ├── cluster_eval_utils.py
    ├── cluster_score_candidates.py    # Score/rank candidates and cache results
    ├── cluster_evaluate_ranked.py      # Evaluate cached ranked results
    ├── cluster_rerank_eval.py          # End-to-end ranking and evaluation
    ├── rerank_model.py                 # Minimal consistency-model inference helpers
    └── local_metrics/                  # BLEU, ROUGE-L, and METEOR implementation
        ├── bleu.py
        ├── rouge.py
        └── meteor.py
```


## Download

The dataset files are stored with Git LFS. Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/bluewhalebug/CoCoRerank.git
cd CoCoRerank
git lfs pull
```

## Environment

The code is written in Python and requires PyTorch. A typical environment can
be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers tqdm numpy scikit-learn nltk wandb rank-bm25 pycocoevalcap==1.2
```

A CUDA-enabled PyTorch installation is recommended for semantic screening,
training, and model-based reranking. `nltk` is needed only for retrieval with
`--nn_type mixed`, and `rank-bm25` is needed for the MBR baseline.

The exact BLEU, ROUGE-L, and METEOR wrappers used in the experiments are
bundled under `evaluate/local_metrics`. Test-mode METEOR requires a Java runtime
available as `java` on `PATH`. Its METEOR 1.5 JAR and English paraphrase table
are supplied by the pinned `pycocoevalcap==1.2` dependency instead of being
duplicated in this repository. The METEOR Java heap limit can be changed with
`METEOR_JAVA_MEMORY` (default: `2G`). Model-based evaluation uses the bundled
`evaluate/rerank_model.py`; `--rerank-module-path` remains available only for
custom overrides.

All commands below assume that the working directory is the repository root:

```bash
cd CoCoRerank
```

## Data format

The repository includes all four partitions of the dataset (86,688
samples): `data/train_set.jsonl` (52,011), `data/retrieval_set.jsonl` (17,337),
`data/val_set.jsonl` (8,670), and `data/test_set.jsonl` (8,670). The training and
validation splits support easy-stage retraining, the retrieval split supplies
in-context examples, and the test split supports evaluation. Hard-stage
training still requires separately generated hard negatives, and model-based
evaluation requires encoder weights, trained checkpoints, and generated
candidates.

Unless otherwise noted, inputs are UTF-8 JSONL files with one object per line.
The main dataset fields are:

```json
{
  "sha": "commit identifier",
  "commit": "code diff",
  "type": "fix",
  "scope": "parser",
  "subject": "handle null values"
}
```

`sha` is optional for training but is recommended for traceability and for
excluding a query from its own retrieval results. `commit` denotes the code
diff rather than the commit hash.

## 1. Dataset filtering

### 1.1 Structural normalization

`dataset_filter/clean.py` removes records with missing `type` or `scope`, drops
excluded/non-semantic types, maps known type variants to canonical labels, and
removes trailing pull-request identifiers such as `(#123)` from subjects.

```bash
python dataset_filter/clean.py \
  --input-path data/parsed_raw_data.jsonl \
  --output-path data/normalized_data.jsonl
```

The input to this script is expected to have already been parsed into `type`,
`scope`, and `subject` fields.

### 1.2 Semantic quality filtering

`dataset_filter/embedding_noise_filter.py` mean-pools a Hugging Face encoder and
computes the cosine similarity between each code diff and its subject. To use
the paper's screening threshold (samples with similarity at or below `0.4` are
flagged for further inspection), run:

```bash
python dataset_filter/embedding_noise_filter.py \
  --input-path data/normalized_data.jsonl \
  --output-dir outputs/semantic_filter \
  --model-path /path/to/Qwen3-0.6B \
  --device cuda:0 \
  --batch-size 1 \
  --threshold 0.4
```

The output directory contains:

- `embedding_scores.jsonl`: a score for every successfully processed sample;
- `suspicious_embedding_noise.jsonl`: full low-scoring input records;
- `suspicious_embedding_noise_sorted.jsonl`: low-scoring records sorted by
  similarity;
- `embedding_errors.jsonl`: failed records when `--continue-on-error` is used;
- `embedding_noise_summary.json`: run configuration and aggregate statistics.

Instead of a fixed threshold, `--top-ratio` or `--top-k` can select the lowest
scoring samples. In the paper's pipeline, low-scoring or unscorable records are
then assessed by an LLM for accuracy, specificity, conciseness, and
completeness using `prompt/quality_assessment.md`. The LLM-serving call is not
included here. As described in the paper, LLM scores above 6 are retained. The
resulting dataset contains 86,688 commit-message pairs and is stratified by
commit type into training, retrieval, validation, and test partitions with a
6:2:1:1 ratio.

## 2. Retrieval for in-context examples

`retrieval/build_icl_index.py` implements the NNGen-style bag-of-words cosine
retrieval used to select five historical examples for each query:

```bash
python retrieval/build_icl_index.py \
  --retrieval_path data/retrieval_set.jsonl \
  --input_path data/test_set.jsonl \
  --output_path outputs/test_icl_cache.jsonl \
  --top_k 5 \
  --nn_type cos \
  --message_mode type_scope_subject \
  --exclude_self \
  --include_meta
```

Each output row has the following form:

```json
{
  "id": "query identifier",
  "sha": "query commit identifier",
  "icl_commit": ["retrieved diff 1", "retrieved diff 2"],
  "icl_msg": ["fix(parser): handle null values", "..."],
  "icl_meta": [{"sha": "...", "index": 0, "cosine": 0.5, "bleu": null, "tokens": 100}]
}
```

Use `--nn_type mixed --search_k K` to rerank cosine-retrieved examples with
sentence BLEU. An optional total token budget can be supplied with
`--max_tokens`; pass a local tokenizer via `--tokenizer_path` for model-specific
token counts.

The LLM candidate-generation prompt is provided in
`prompt/candidate_generation.md`. The subject-only variant is provided in
`prompt/subject_only.md`. In the experiments, the generator produces 100
CCS-compliant candidates per test instance.

## 3. Train the consistency scorer

Training has two stages. Configuration is supplied through environment
variables, and Weights & Biases logging defaults to offline mode.

### 3.1 Easy stage

The easy stage learns structured context-to-subject alignment with an in-batch
multi-positive contrastive objective:

```bash
MODEL_NAME=/path/to/encoder \
TRAIN_PATH=data/train_set.jsonl \
VAL_PATH=data/val_set.jsonl \
SAVE_PATH=checkpoints/easy.pt \
WANDB_DIR=wandb_runs \
WANDB_MODE=offline \
DEVICE=cuda:0 \
python train/easytrain.py
```

The defaults used by the script match the main easy-stage settings in the
paper: batch size 96, 8 epochs, learning rate `2e-5`, temperature `0.05`, warmup
ratio `0.06`, and random seed 42. Useful overrides include `BATCH_SIZE`,
`EPOCHS`, `LR`, `TEMP`, `POOLING`, `COMMIT_MAX_LEN`, `META_MAX_LEN`,
`SUBJECT_MAX_LEN`, and `NUM_WORKERS`.

### 3.2 Hard stage

Hard-stage rows extend the base format with LLM-generated hard negatives:

```json
{
  "commit": "code diff",
  "type": "fix",
  "scope": "parser",
  "subject": "handle null values",
  "hardneg": [
    "ignore null values",
    "handle empty configuration"
  ]
}
```

The hard-negative generation prompt is provided in `prompt/hard_negative.md`;
its LLM-serving code is not included here. After generating the hard-negative
training and validation files, continue from the best easy-stage checkpoint:

```bash
MODEL_NAME=/path/to/encoder \
TRAIN_PATH=data/hard_train.jsonl \
VAL_PATH=data/hard_validation.jsonl \
PRETRAIN_PATH=checkpoints/easy.pt \
SAVE_PATH=checkpoints/hard.pt \
WANDB_DIR=wandb_runs \
WANDB_MODE=offline \
DEVICE=cuda:0 \
python train/hardtrain.py
```

The hard-stage defaults are batch size 48, 5 epochs, learning rate `1e-5`,
temperature `0.03`, at most 6 hard negatives per instance (`NEG_K=6`),
hard-negative loss weight `1.5`, warmup ratio `0.06`, and random seed 42.

## 4. Candidate reranking and evaluation

The candidate file and test file must contain the same number of rows in the
same order. A candidate row contains generated CCS messages:

```json
{
  "index": 0,
  "commit_message": [
    "fix(parser): handle null values",
    "fix(parser): avoid null pointer errors"
  ]
}
```

Model-based rankers additionally require a vote-metadata file with the
consensus type and scope for each row:

```json
{
  "index": 0,
  "most_common_type": "fix",
  "most_common_scope": "parser"
}
```

The following rankers are supported:

- `hard_model`: the full consistency scorer after hard-negative training;
- `easy_model`: the **w/o Hard Negatives** ablation;
- `mbr`: BM25-based minimum Bayes risk baseline;
- `vote`: weak-match cluster voting used for the **w/o Horizontal
  Consistency** ablation;
- `Majority_Voting`: normalized exact-match majority-voting baseline.

The command-line ranker names are case-sensitive. The paper settings map to
the implementation as follows:

| Paper setting | Ranker | Evaluation option |
|---|---|---|
| CoCoRerank | `hard_model` | add `--cluster-aware-eval` |
| w/o Hard Negatives | `easy_model` | add `--cluster-aware-eval` |
| w/o Horizontal Consistency | `vote` | add `--cluster-aware-eval` |
| w/o Vertical Consistency | `hard_model` | omit `--cluster-aware-eval` |
| MBR | `mbr` | not applicable |
| Majority Voting | `Majority_Voting` | not applicable |

Here, `--cluster-aware-eval` enables weak-match cluster aggregation and the
representative ordering used for vertical consistency. Omitting it evaluates
the global individual-score order and therefore implements the **w/o Vertical
Consistency** ablation.

### Model-based reranking

For learned rankers, the bundled `evaluate/rerank_model.py` provides
`StructuredBiEncoder` and all checkpoint, configuration, and tokenization
helpers needed by `evaluate/cluster_rerank_eval.py`. The scripts select this
module by default. `--rerank-module-path` is only needed to override it with a
different compatible implementation.

Candidate scoring can be run once and cached:

```bash
python evaluate/cluster_score_candidates.py \
  --cand-path outputs/candidates.jsonl \
  --vote-meta-path outputs/vote_metadata.jsonl \
  --test-path data/test_set.jsonl \
  --rankers easy_model hard_model mbr vote Majority_Voting \
  --model-path easy_model=checkpoints/easy.pt \
  --model-path hard_model=checkpoints/hard.pt \
  --output-dir outputs/ranked_results \
  --device cuda:0
```

To run only the model-free rankers, omit the vote metadata, model paths, and
inference module:

```bash
python evaluate/cluster_score_candidates.py \
  --cand-path outputs/candidates.jsonl \
  --test-path data/test_set.jsonl \
  --rankers mbr vote Majority_Voting \
  --output-dir outputs/baseline_ranked_results
```

### Evaluate cached rankings

Cached rankings can be reevaluated without rerunning the encoders. This is the
recommended entry point for changing Pass@k, BLEU order, or hit threshold:

```bash
python evaluate/cluster_evaluate_ranked.py \
  --ranked-dir outputs/ranked_results \
  --rankers easy_model hard_model mbr vote Majority_Voting \
  --summary-output results/summary.json \
  --ks 1,2,3,5 \
  --bleu-order 1 \
  --hit-threshold 70 \
  --cluster-aware-eval
```

`--cluster-aware-eval` evaluates the representative ordering induced by weak
subject clusters and thus includes vertical consistency. Without this flag,
the scripts evaluate the global candidate-score order, corresponding to the
**w/o Vertical Consistency** ablation for `hard_model`. The summary includes
Pass@k and top-1 subject metrics.

Raw, unreranked candidates can also be evaluated:

```bash
python evaluate/cluster_evaluate_ranked.py \
  --skip-ranked \
  --include-raw-cand-pass \
  --include-raw-type-pass \
  --raw-cand-path outputs/candidates.jsonl \
  --test-path data/test_set.jsonl \
  --summary-output results/raw_summary.json \
  --ks 1,2,3,5 \
  --hit-threshold 70
```

Alternatively, `evaluate/cluster_rerank_eval.py` combines scoring and metric
calculation in one command and accepts the union of the scoring and evaluation
arguments shown above.

## Reproducibility notes

- The scripts use a default seed of 42 for both training stages.
- Training checkpoints store the model state, optimizer and scheduler states,
  configuration, epoch, validation loss, and type vocabulary.
- Input JSONL files are aligned by row; use `--strict-index` during reranking to
  detect mismatched explicit indices.
- The paper uses weak-match subject clustering, cluster-level score aggregation,
  and one representative per cluster for the final ranked list.
- The metric wrappers are included under `evaluate/local_metrics`. The METEOR
  1.5 runtime assets are supplied by the pinned `pycocoevalcap==1.2`
  dependency.
- All four dataset splits are included. Pretrained model weights, trained
  checkpoints, generated candidates, hard negatives, and LLM-serving code are
  not included in this repository.
