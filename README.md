 # Context-Aware Hate Speech Detection Using Conversational Threads

Binary stereotype classification on Spanish tweets/comments using conversational thread context. We compare a context-free baseline against three context-aware architectures built on top of XLM-RoBERTa and study how much the surrounding thread (parent and root tweets/comments) helps detect stereotypes.

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

Given a Spanish tweet/comment and its conversational thread (parent tweet/comment + root tweet/comment), predict whether the text contains a stereotype (label 1) or not (label 0).

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

Thread context is resolved by mapping parent and root tweet IDs to their actual text using the full dataset as a lookup. Samples where parent or root text could not be resolved are dropped. Only samples with both parent and root context are kept for training and evaluation, ensuring all models are compared on the same population.

---

## Models

### Baseline
Standard XLM-RoBERTa fine-tuned on tweet text only, with no thread context. Serves as the reference point.

### Hierarchical
Encodes target tweet, parent, and root separately with a **shared** XLM-RoBERTa encoder. The three CLS representations are combined via multi-head self-attention with learned position embeddings for thread order. The tweet is the query; root and parent are keys/values.

### Cross-Attention
Token-level context fusion. Tweet tokens attend to the full token sequence of the concatenated context (root + parent) via stacked cross-attention layers. Captures fine-grained lexical interactions between the tweet and its thread.

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
│   ├── visualize_loss.py             # loss curves  
│   ├── attention_report.py           # Per-example attention weight report
│   ├── integrated_gradient.py        # IG-based token attribution
│   ├── interpretability_extended.py  # Thread-level and token-level attention distributions
│   ├── top_tokens.py                 # Most attended context tokens
│   ├── pick_index.py                 # Sample selection for qualitative analysis
│   └── print_test_samples.py         # Browse test set to stdout
└── utils/
    ├── imbalance.py                  # Class weights and oversampling
    └── inference_utils.py            
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

Runs inference on the full test set and extracts attention weights from the hierarchical and cross-attention models. Produces bar charts of mean attention weights per thread position split by class, violin plots of attention weight distributions, cross-attention segment importance (fraction of attention on root vs parent, using layer 1 weights) and a layer 1 vs layer 2 comparison.

`attention_report.py`: Per-example attention report

Generates a plain-text report combining predictions and attention weights from all three models for individual test samples. For the hierarchical model, reports thread position weights (root / parent / target). Use `--tweet_indices` to inspect specific samples by index.

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

Uses [Captum](https://captum.ai/) to attribute each model's prediction to individual tweet tokens via Integrated Gradients. Context (root + parent) is held fixed while tweet token embeddings are interpolated from a zero baseline. For each selected test case, outputs a side-by-side bar chart of the top-20 tokens by attribution score across the baseline, hierarchical and cross-attention models. Cases can be selected automatically from `inference_per_sample.csv` by verdict, or targeted by index with `--tweet_idx`.

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

## Figures

Generated figures from all analysis scripts are available in `figures/`:

- **`model_comparison.png`** — bar charts and validation curves comparing all four models across seeds (mean ± std)
- **`loss_curves.png`** — training and validation loss curves per model with ± std band across seeds
- **`attention_by_class_hierarchical.png`** / **`attention_by_class_cross-attention.png`** — mean attention weights split by class (stereotype vs non-stereotype)
- **`attention_distribution_hierarchical.png`** / **`attention_distribution_cross-attention.png`** — violin plots of attention weight distributions across all test samples
- **`cross_attention_layer_comparison.png`** — layer 1 vs layer 2 attention shift in the cross-attention model
- **`cross_attn_top_tokens.png`** / **`cross_attn_top_tokens.csv`** — most attended context tokens ranked by discriminative score, split by class
- **`ig_comparison_case_{index}*.png`** — Integrated Gradients attribution plots for selected test cases, comparing token-level attributions across all three models side by side. Cases include both context-win samples (where context resolves implicit stereotyping) and baseline-win samples (where context misleads the model)
- **`attention_report.txt`** — per-example attention weights and predictions for all models across the test set

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

---

## References

- Schmeisser-Nieto et al. (2025) — [StereoHoax: A Dataset for Stereotype Detection in Spanish Social Media](https://arxiv.org/abs/2501.xxxxx)
- Schmeisser-Nieto et al. (2024) — [DETESTS-Dis: A Dataset for Stereotype Detection in Spanish](https://arxiv.org/abs/2401.xxxxx)
- Beddiar et al. (2021) — [Data Augmentation for Hate Speech Detection via Back-Translation](https://arxiv.org/abs/2101.xxxxx)
- Jain & Wallace (2019) — [Attention is not Explanation](https://arxiv.org/abs/1902.10186)
- Sundararajan et al. (2017) — [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)