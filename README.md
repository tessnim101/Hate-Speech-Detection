 # Context-Aware Hate Speech Detection Using Conversational Threads

Binary stereotype classification on Spanish social media posts using conversational thread context. We compare a context-free baseline against three context-aware architectures built on top of XLM-RoBERTa and study how much the surrounding thread (parent and root posts) helps detect stereotypes.

---

## Quick Start

### Option 1: Docker (EPFL RCP)
Followed the provided RCP tutorial to build and run the Docker image on the cluster.

### Option 2: Local

```bash
pip install -r requirements.txt

python main.py \
    --dataset_path "data/spanish_subset_collapsed/" \
    --results_path "results/"
```

---

## Task

Given a Spanish post and its conversational thread (parent post + root post), predict whether the post contains a stereotype (label 1) or not (label 0).

Each sample includes:
- `text` — the target tweet/comment
- `parent_text` — the directly preceding tweet/comment in the thread
- `root_text` — the first tweet/comment in the thread

---

## Data

Download the dataset from HuggingFace and place in `data/spanish_subset_collapsed/`:

- [DETESTS-Dis](https://huggingface.co/datasets/CLiC-UB/DETESTS-Dis/tree/main)

The dataset combines two sources identifiable via the `source` column:
- **StereoHoax** — Twitter/X conversational threads reacting to racial hoaxes
- **DETESTS** — Spanish newspaper comments

Expected files:
```
data/spanish_subset_collapsed/
├── train.csv
└── test.csv
```

### Preprocessing

All text fields (`text`, `parent_text`, `root_text`) are cleaned before training and inference:
- Lowercased
- URLs and mentions (`@user`) removed
- Hashtag symbols removed (word kept)
- Punctuation removed
- Whitespace collapsed

Thread context is resolved by mapping parent and root post IDs to their actual text using the full dataset as a lookup. Samples where parent or root text could not be resolved are dropped. Only samples with both parent and root context are kept for training and evaluation, ensuring all models are compared on the same population.

---

## Models

### Baseline
Standard XLM-RoBERTa fine-tuned on post text only, with no thread context. Serves as the reference point.

### Hierarchical
Encodes post, parent, and root separately with a **shared** XLM-RoBERTa encoder. The three CLS representations are combined via multi-head self-attention with learned position embeddings for thread order. The post is the query; root and parent are keys/values.

### Cross-Attention
Token-level context fusion. Post tokens attend to the full token sequence of the concatenated context (root + parent) via stacked cross-attention layers. Captures fine-grained lexical interactions between the post and its thread.

### BT-Augmented
Same architecture as Hierarchical, but trained on a back-translation-augmented dataset — training samples are paraphrased via Spanish → English → Spanish translation to expand the training set.

---

## Project Structure

```
.
├── main.py                           # Training entry point (all four models, multi-seed)
├── inference.py                      # Evaluate saved models and compare predictions
├── config.py                         # Hyperparameters and model config
├── requirements.txt                  # Python dependencies
├── data/
│   ├── loader.py                     # Load train/test CSV + back-translation file
│   ├── preprocessing.py              # Text cleaning, tokenization, train/val split
│   ├── translate.py                  # Back-translation script
│   └── inspect_data.py               # Data exploration utilities
├── modeling/
│   └── models.py                     # HierarchicalContextModel, CrossAttentionContextModel
├── training/
│   ├── runners.py                    # run_baseline / run_hierarchical / run_cross_attention / run_augmented
│   ├── trainer_utils.py              # HuggingFace Trainer setup, compute_metrics
│   ├── helpers.py                    # Callbacks, dataset helpers, result saving
│   └── metrics.py                    # Metric utilities
├── analysis/
│   ├── visualize.py                  # Plot F1/accuracy across seeds and models
│   ├── attention_report.py           # Per-example attention weight report
│   ├── integrated_gradient.py        # IG-based token attribution
│   ├── interpretability_extended.py  # Thread-level and token-level attention distributions
│   ├── top_tokens.py                 # Most attended context tokens
│   ├── pick_index.py                 # Sample selection for qualitative analysis
│   └── print_test_samples.py         # Browse test set to stdout
└── utils/
    ├── imbalance.py                  # Class weights and oversampling
    └── inference_utils.py            # Encoding helpers, model loaders
```

---

## Usage

### Training

Trains all four models across three seeds (42, 123, 456) and saves the median-seed checkpoint for each.

```bash
python main.py \
    --dataset_path "data/spanish_subset_collapsed/" \
    --results_path "results/" \
    --imbalance_strategy "class_weights"   # or "oversample" or "none"
```

Results are saved under `results/results_<model>/seed_<N>/` per seed, and aggregated into:
- `results/all_test_results.csv` — per-seed, per-model metrics
- `results/summary_mean_std.csv` — mean ± std across seeds
- `results/best_model_<name>/` — median-seed checkpoint for each model

### Inference

Evaluate and compare saved models on the test set:

```bash
python inference.py \
    --dataset_path          "data/spanish_subset_collapsed/" \
    --baseline_path         "results/best_model_baseline/" \
    --context_path          "results/best_model_hierarchical/" \
    --cross_attention_path  "results/best_model_cross_attention/" \
    --augmented_path        "results/best_model_augmented/" \
    --results_path          "results/"
```

Outputs per-sample predictions (`inference_per_sample.csv`) and a summary with deltas vs. baseline.

### Back-Translation Augmentation

```bash
python data/translate.py
```

Generates `train_bt.csv` in your dataset folder. Must be run before training the augmented model.

### Visualization

```bash
python analysis/visualize.py \
    --results_path                 "results/" \
    --results_path_baseline        "results/results_baseline/" \
    --results_path_hierarchical    "results/results_hierarchical/" \
    --results_path_cross_attention "results/results_cross_attention/" \
    --results_path_augmented       "results/results_augmented/" \
    --output_path                  "figures/"
```

The four plots are:
- **Top-left** — grouped bar chart of F1 Macro, Accuracy, F1 Class 0, F1 Class 1 (mean ± std)
- **Top-right** — per-class F1 side by side per model
- **Bottom-left** — validation F1 Macro per epoch with ± std band
- **Bottom-right** — validation F1 Class 1 per epoch

---

## Interpretation & Analysis

All scripts in `analysis/` are standalone. Each file has its full run command at the top.

`interpretability_extended.py`: Attention distributions

Thread-level attention weights for Hierarchical and BT-Augmented and context segment importance for Cross-Attention, split by class (stereotype vs non-stereotype).

`attention_report.py`: Per-example attention report

Plain-text report with per-example predictions and attention weights for all models. Use `--tweet_indices` to inspect specific samples.

`pick_index.py`: Sample selection for qualitative analysis

Browse `inference_per_sample.csv` by verdict, label, and confidence threshold to find interesting cases for IG or attention analysis.

```bash
python analysis/pick_index.py \
    --per_sample     "results/inference_per_sample.csv" \
    --dataset_path   "data/spanish_subset_collapsed/" \
    --verdict        context_wins \
    --label          1 \
    --min_confidence 0.6 \
    --n              10
```

`print_test_samples.py`: Browse test set

```bash
python analysis/print_test_samples.py \
    --dataset_path "data/spanish_subset_collapsed/" \
    --label 1   # stereotype only
```

`integrated_gradient.py`: Token attribution via Integrated Gradients

Attributes predictions to individual post tokens using [Captum](https://captum.ai/). Outputs bar charts comparing token attributions across models for selected test cases.

```bash
python analysis/integrated_gradient.py \
    --dataset_path         "data/spanish_subset_collapsed/" \
    --baseline_path        "results/best_model_baseline/" \
    --context_path         "results/best_model_hierarchical/" \
    --cross_attention_path "results/best_model_cross_attention/" \
    --results_path         "results/figures/" \
    --per_sample           "results/inference_per_sample.csv" \
    --ig_steps             50 \
    --tweet_idx            102
```

`top_tokens.py`: Most attended context tokens

Aggregates cross-attention weights over the test set to investigate which context tokens are most predictive of stereotype vs non-stereotype, ranked by discriminative score.

```bash
python analysis/top_tokens.py \
    --dataset_path         "data/spanish_subset_collapsed/" \
    --cross_attention_path "results/best_model_cross_attention/" \
    --results_path         "results/figures/" \
    --top_k                30
```

---

## Configuration

All hyperparameters are in [config.py](config.py):

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `xlm-roberta-base` | Backbone model |
| `max_len` | 192 | Max tokens for baseline / hierarchical |
| `max_len_tweet` | 128 | Max tweet tokens for cross-attention |
| `max_len_context` | 256 | Max context tokens for cross-attention |
| `learning_rate` | 1e-5 | AdamW learning rate |
| `epochs` | 7 | Max training epochs |
| `early_stopping_patience` | 3 | Early stopping patience |
| `train_bs` / `eval_bs` | 8 | Batch sizes |