"""
Extended interpretability analysis.

Complements interpretability.py (which covers hierarchical thread-level attention).
Covers:
  1. Cross-attention token-level segment importance (hoax vs root vs parent)
  2. Per-sample cross-attention heatmaps on case studies
  3. Integrated Gradients on tweet tokens across all 3 models
  4. Cross-model token importance comparison on case studies

python token_attention.py \
    --dataset_path          data/spanish_subset/ \
    --baseline_path         results/best_model_baseline/ \
    --context_path          results/best_model_context/ \
    --cross_attention_path  results/best_model_cross_attention/ \
    --per_sample            results/inference_per_sample.csv \
    --results_path          results/figures/ \
    --n_cases               4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text
from modeling.models import HierarchicalContextModel, CrossAttentionHoaxModel
from config import CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",         required=True)
    p.add_argument("--baseline_path",        required=True)
    p.add_argument("--context_path",         required=True)
    p.add_argument("--cross_attention_path", required=True)
    p.add_argument("--per_sample",           required=True,
                   help="Path to inference_per_sample.csv")
    p.add_argument("--results_path",         required=True)
    p.add_argument("--batch_size",           type=int, default=32)
    p.add_argument("--n_cases",              type=int, default=4,
                   help="Case studies per verdict category")
    p.add_argument("--ig_steps",             type=int, default=50,
                   help="IG interpolation steps (higher = more accurate, slower)")
    return p.parse_args()


def load_baseline(path, device):
    model     = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def load_hierarchical(path, device):
    model   = HierarchicalContextModel(CONFIG["model_name"]).to(device)
    weights = load_file(str(Path(path) / "model.safetensors"), device=str(device))
    model.load_state_dict(weights)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def load_cross_attention(path, device):
    model   = CrossAttentionHoaxModel(CONFIG["model_name"]).to(device)
    weights = load_file(str(Path(path) / "model.safetensors"), device=str(device))
    model.load_state_dict(weights)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def enc(tokenizer, texts, max_len, device):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)


def build_context_string(hoax, root, parent):
    """
    Build the context string and return both the full string and the
    individual part strings (for segment boundary detection).
    """
    parts = {}
    segs  = []
    if hoax   and str(hoax).strip():
        parts["hoax"]   = f"Hoax: {hoax}"
        segs.append(parts["hoax"])
    if root   and str(root).strip():
        parts["root"]   = f"Thread: {root}"
        segs.append(parts["root"])
    if parent and str(parent).strip():
        parts["parent"] = f"Reply to: {parent}"
        segs.append(parts["parent"])
    return " | ".join(segs), parts


def get_segment_masks(tokenizer, parts: dict, ctx_ids: list[int]):
    """
    For each segment (hoax/root/parent), return a boolean mask over
    ctx_ids indicating which positions belong to that segment.
    Uses sliding window to find token subsequences.
    """
    masks = {name: np.zeros(len(ctx_ids), dtype=bool) for name in ["hoax", "root", "parent"]}
    for name, text in parts.items():
        seg_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(len(ctx_ids) - len(seg_ids) + 1):
            if ctx_ids[start : start + len(seg_ids)] == seg_ids:
                masks[name][start : start + len(seg_ids)] = True
                break
    return masks


@torch.no_grad()
def extract_cross_attention_weights(model, tokenizer, df, batch_size, device):
    """
    Run inference over df and collect per-sample attention weights.

    Returns:
        all_weights:  list of (num_heads, tweet_len, ctx_len) arrays
        all_ctx_ids:  list of context token-id lists
        all_twt_ids:  list of tweet token-id lists
        all_parts:    list of {hoax/root/parent -> segment string} dicts
    """
    all_weights, all_ctx_ids, all_twt_ids, all_parts = [], [], [], []

    texts        = df["text"].tolist()
    hoax_texts   = df["hoax"].tolist()
    root_texts   = df["root_text"].tolist()
    parent_texts = df["parent_text"].tolist()

    for i in range(0, len(texts), batch_size):
        h_b = hoax_texts[i   : i + batch_size]
        r_b = root_texts[i   : i + batch_size]
        p_b = parent_texts[i : i + batch_size]
        t_b = texts[i        : i + batch_size]

        contexts, parts_batch = [], []
        for h, r, p in zip(h_b, r_b, p_b):
            ctx_str, parts = build_context_string(h, r, p)
            contexts.append(ctx_str)
            parts_batch.append(parts)

        ctx_enc = enc(tokenizer, contexts, CONFIG["max_len_context"], device)
        twt_enc = enc(tokenizer, t_b,      CONFIG["max_len_tweet"],   device)

        out = model(
            context_input_ids=      ctx_enc["input_ids"],
            context_attention_mask= ctx_enc["attention_mask"],
            tweet_input_ids=        twt_enc["input_ids"],
            tweet_attention_mask=   twt_enc["attention_mask"],
            return_attention_weights=True,
        )

        # (B, num_heads, tweet_len, ctx_len)
        weights = out["attention_weights"].cpu().numpy()

        for j in range(len(t_b)):
            all_weights.append(weights[j])
            all_ctx_ids.append(ctx_enc["input_ids"][j].cpu().tolist())
            all_twt_ids.append(twt_enc["input_ids"][j].cpu().tolist())
            all_parts.append(parts_batch[j])

    return all_weights, all_ctx_ids, all_twt_ids, all_parts


def compute_segment_importance(all_weights, all_ctx_ids, all_parts, tokenizer):
    """
    For each sample compute how much total attention flows to each
    context segment (hoax / root / parent), normalised to sum to 1.

    Returns a DataFrame with columns [hoax, root, parent, sample_idx].
    """
    rows = []
    for idx, (weights, ctx_ids, parts) in enumerate(zip(all_weights, all_ctx_ids, all_parts)):
        # Average across heads and tweet query positions → (ctx_len,)
        avg = weights.mean(axis=0).mean(axis=0)

        masks = get_segment_masks(tokenizer, parts, ctx_ids)
        total = avg.sum() + 1e-9
        row   = {"sample_idx": idx}
        for name in ["hoax", "root", "parent"]:
            row[name] = float(avg[masks[name]].sum() / total)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_segment_importance(seg_df, labels, out_dir):
    """
    Bar chart: mean normalised attention to each context segment, by class.
    """
    seg_df         = seg_df.copy()
    seg_df["label"] = labels

    segments    = ["hoax", "root", "parent"]
    seg_labels  = {"hoax": "Hoax description", "root": "Root tweet", "parent": "Parent tweet"}
    seg_colors  = {"hoax": "#4C72B0", "root": "#DD8452", "parent": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, (lbl, name) in zip(axes, [(0, "Not stereo (class 0)"), (1, "Stereo (class 1)")]):
        sub   = seg_df[seg_df["label"] == lbl]
        means = [sub[s].mean() for s in segments]
        stds  = [sub[s].std()  for s in segments]

        bars = ax.bar(
            [seg_labels[s] for s in segments], means,
            yerr=stds, capsize=5,
            color=[seg_colors[s] for s in segments], alpha=0.85,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalised attention" if lbl == 0 else "")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                    f"{mean:.2f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Cross-Attention: Context Segment Importance by Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(out_dir) / "cross_attn_segment_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()


def plot_attention_heatmap(weights, ctx_tokens, twt_tokens, title, out_path):
    """
    Heatmap of tweet tokens (rows) × context tokens (cols),
    averaged across attention heads.
    """
    avg = weights.mean(axis=0)  # (tweet_len, ctx_len)

    # Strip padding and special tokens
    skip       = {"<pad>", "<s>", "</s>"}
    twt_clean  = [(i, t) for i, t in enumerate(twt_tokens) if t not in skip]
    ctx_clean  = [(j, t) for j, t in enumerate(ctx_tokens) if t not in skip]

    if not twt_clean or not ctx_clean:
        return

    twt_idx, twt_labels = zip(*twt_clean)
    ctx_idx, ctx_labels = zip(*ctx_clean)
    sub = avg[np.ix_(list(twt_idx), list(ctx_idx))]

    fig, ax = plt.subplots(figsize=(
        max(10, len(ctx_labels) * 0.35),
        max(4,  len(twt_labels) * 0.3),
    ))
    im = ax.imshow(sub, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(ctx_labels)))
    ax.set_xticklabels(ctx_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(twt_labels)))
    ax.set_yticklabels(twt_labels, fontsize=8)
    ax.set_xlabel("Context tokens", fontsize=10)
    ax.set_ylabel("Tweet tokens",   fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close()


class BaselineIGWrapper(nn.Module):
    """Accepts tweet embeddings directly for captum compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_embeds, attention_mask):
        out = self.model.roberta(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        return self.model.classifier(out.last_hidden_state[:, 0, :])


class HierarchicalIGWrapper(nn.Module):
    """IG on tweet embeddings; root and parent IDs are held fixed."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tweet_embeds, tweet_mask,
                root_ids, root_mask, parent_ids, parent_mask):
        root_cls   = self.model.encode(root_ids,   root_mask)
        parent_cls = self.model.encode(parent_ids, parent_mask)

        tweet_out = self.model.encoder(
            inputs_embeds=tweet_embeds,
            attention_mask=tweet_mask,
        )
        tweet_cls = tweet_out.last_hidden_state[:, 0, :]

        positions = torch.arange(3, device=tweet_cls.device)
        pos_emb   = self.model.position_embeddings(positions).unsqueeze(0)
        thread    = torch.stack([root_cls, parent_cls, tweet_cls], dim=1) + pos_emb

        root_empty   = (root_mask.sum(dim=1)   <= 2)
        parent_empty = (parent_mask.sum(dim=1) <= 2)
        tweet_empty  = torch.zeros(
            root_empty.shape[0], dtype=torch.bool, device=tweet_cls.device
        )
        key_padding_mask = torch.stack([root_empty, parent_empty, tweet_empty], dim=1)

        attn_out, _ = self.model.thread_attention(
            query=thread[:, 2:, :],
            key=thread, value=thread,
            key_padding_mask=key_padding_mask,
        )
        return self.model.classifier(attn_out.squeeze(1))


class CrossAttentionIGWrapper(nn.Module):
    """IG on tweet embeddings; context IDs are held fixed."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tweet_embeds, tweet_mask, ctx_ids, ctx_mask):
        ctx_tokens = self.model.encode(ctx_ids, ctx_mask)

        tweet_out    = self.model.encoder(
            inputs_embeds=tweet_embeds,
            attention_mask=tweet_mask,
        )
        tweet_tokens = tweet_out.last_hidden_state

        ctx_key_padding_mask = (ctx_mask == 0)
        attended, _  = self.model.cross_attention(
            query=tweet_tokens,
            key=ctx_tokens, value=ctx_tokens,
            key_padding_mask=ctx_key_padding_mask,
        )
        enriched = self.model.layer_norm(tweet_tokens + attended)
        return self.model.classifier(enriched[:, 0, :])


# IG helpers
def get_embeddings(model, model_type, input_ids):
    """
    Extract word embeddings from the model's embedding layer.
    Baseline uses .roberta, others use .encoder.
    """
    with torch.no_grad():
        if model_type == "baseline":
            return model.roberta.embeddings.word_embeddings(input_ids)
        else:
            return model.encoder.embeddings.word_embeddings(input_ids)


def compute_ig(wrapper, input_embeds, additional_args, target_class, n_steps):
    """
    Run Integrated Gradients on input_embeds.

    Returns:
        (seq_len,) numpy array of per-token importance (summed across embed dim)
    """
    ig       = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(input_embeds)
    attrs, _ = ig.attribute(
        input_embeds,
        baselines=baseline,
        target=target_class,
        additional_forward_args=additional_args,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()


# IG case studies
def run_ig_case_studies(
    baseline_model,  baseline_tokenizer,
    hier_model,      hier_tokenizer,
    ca_model,        ca_tokenizer,
    cases_df, device, n_steps,
):
    """
    Run IG on tweet tokens for all 3 models on each case study row.

    Returns list of result dicts with tokens and importance per model.
    """
    b_wrapper  = BaselineIGWrapper(baseline_model).to(device).eval()
    h_wrapper  = HierarchicalIGWrapper(hier_model).to(device).eval()
    ca_wrapper = CrossAttentionIGWrapper(ca_model).to(device).eval()

    results = []

    for _, row in cases_df.iterrows():
        tweet   = row["text"]
        label   = int(row["stereotype"])
        verdict = row.get("verdict", "")

        # baseline
        b_enc    = enc(baseline_tokenizer, [tweet], CONFIG["max_len"], device)
        b_embeds = get_embeddings(baseline_model, "baseline", b_enc["input_ids"])
        b_embeds = b_embeds.detach().requires_grad_(True)

        b_imp    = compute_ig(
            b_wrapper, b_embeds,
            additional_args=(b_enc["attention_mask"],),
            target_class=label, n_steps=n_steps,
        )
        b_tokens = baseline_tokenizer.convert_ids_to_tokens(b_enc["input_ids"][0].tolist())

        # hierarchical
        h_enc      = enc(hier_tokenizer, [tweet],                    CONFIG["max_len"], device)
        root_enc   = enc(hier_tokenizer, [row.get("root_text",   "")], CONFIG["max_len"], device)
        parent_enc = enc(hier_tokenizer, [row.get("parent_text", "")], CONFIG["max_len"], device)

        h_embeds = get_embeddings(hier_model, "hierarchical", h_enc["input_ids"])
        h_embeds = h_embeds.detach().requires_grad_(True)

        h_imp    = compute_ig(
            h_wrapper, h_embeds,
            additional_args=(
                h_enc["attention_mask"],
                root_enc["input_ids"],   root_enc["attention_mask"],
                parent_enc["input_ids"], parent_enc["attention_mask"],
            ),
            target_class=label, n_steps=n_steps,
        )
        h_tokens = hier_tokenizer.convert_ids_to_tokens(h_enc["input_ids"][0].tolist())

        # cross-attention
        _, parts   = build_context_string(
            row.get("hoax", ""), row.get("root_text", ""), row.get("parent_text", "")
        )
        ctx_str    = " | ".join(parts.values())
        ca_twt_enc = enc(ca_tokenizer, [tweet],   CONFIG["max_len_tweet"],   device)
        ca_ctx_enc = enc(ca_tokenizer, [ctx_str], CONFIG["max_len_context"], device)

        ca_embeds  = get_embeddings(ca_model, "cross_attention", ca_twt_enc["input_ids"])
        ca_embeds  = ca_embeds.detach().requires_grad_(True)

        ca_imp     = compute_ig(
            ca_wrapper, ca_embeds,
            additional_args=(
                ca_twt_enc["attention_mask"],
                ca_ctx_enc["input_ids"],
                ca_ctx_enc["attention_mask"],
            ),
            target_class=label, n_steps=n_steps,
        )
        ca_tokens  = ca_tokenizer.convert_ids_to_tokens(ca_twt_enc["input_ids"][0].tolist())

        results.append({
            "tweet":   tweet,
            "label":   label,
            "verdict": verdict,
            "baseline":        {"tokens": b_tokens,  "importance": b_imp},
            "hierarchical":    {"tokens": h_tokens,  "importance": h_imp},
            "cross_attention": {"tokens": ca_tokens, "importance": ca_imp},
        })

    return results


def plot_ig_comparison(ig_results, out_dir):
    """
    For each case study: side-by-side horizontal bar charts of tweet
    token importance across all 3 models.
    Positive scores push toward predicted class, negative away.
    """
    model_keys   = ["baseline", "hierarchical", "cross_attention"]
    model_labels = ["Baseline", "Hierarchical", "Cross-Attention"]
    model_colors = ["#6C8EBF", "#D4763B", "#55A868"]
    skip         = {"<s>", "</s>", "<pad>"}

    for idx, result in enumerate(ig_results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"[{result['verdict']}]  label={result['label']}  |  "
            f"{result['tweet'][:90]}{'...' if len(result['tweet']) > 90 else ''}",
            fontsize=9, fontweight="bold",
        )

        for ax, key, mlabel, color in zip(axes, model_keys, model_labels, model_colors):
            tokens     = result[key]["tokens"]
            importance = result[key]["importance"]

            filtered = [(t, s) for t, s in zip(tokens, importance) if t not in skip]
            if not filtered:
                continue
            tokens_f, scores_f = zip(*filtered)
            scores_f = np.array(scores_f)

            # Normalise to [-1, 1]
            max_abs    = np.abs(scores_f).max() + 1e-9
            scores_norm = scores_f / max_abs

            bar_colors = [color if s >= 0 else "#e74c3c" for s in scores_norm]
            y_pos      = np.arange(len(tokens_f))

            ax.barh(y_pos, scores_norm, color=bar_colors, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens_f, fontsize=8)
            ax.set_xlabel("Normalised IG score", fontsize=9)
            ax.set_title(mlabel, fontsize=11, fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlim(-1.1, 1.1)
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        out = Path(out_dir) / f"ig_comparison_case_{idx}_{result['verdict']}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.close()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out_dir = Path(args.results_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, df_test = load_data(args.dataset_path)
    df_test    = filter_contextual_tweets(df_test)
    df_test    = ids_to_text(df_test.copy())
    labels     = df_test["stereotype"].values

    # Models 
    print("[loading] Baseline...")
    baseline_model, baseline_tokenizer = load_baseline(args.baseline_path, device)

    print("[loading] Hierarchical...")
    hier_model, hier_tokenizer = load_hierarchical(args.context_path, device)

    print("[loading] Cross-attention...")
    ca_model, ca_tokenizer = load_cross_attention(args.cross_attention_path, device)

    # Cross-attention segment importance
    print("\n[cross-attention] Extracting attention weights over test set...")
    all_weights, all_ctx_ids, all_twt_ids, all_parts = extract_cross_attention_weights(
        ca_model, ca_tokenizer, df_test, args.batch_size, device,
    )

    seg_df = compute_segment_importance(all_weights, all_ctx_ids, all_parts, ca_tokenizer)
    seg_df["label"] = labels

    print("\n===== Cross-Attention Segment Importance =====")
    for seg in ["hoax", "root", "parent"]:
        print(f"  {seg:<8}  mean={seg_df[seg].mean():.3f}  std={seg_df[seg].std():.3f}")
    print("  --- by class ---")
    for lbl, name in [(0, "Not stereo"), (1, "Stereo")]:
        sub = seg_df[seg_df["label"] == lbl]
        print(f"  {name}:")
        for seg in ["hoax", "root", "parent"]:
            print(f"    {seg:<8}  {sub[seg].mean():.3f}")

    plot_segment_importance(seg_df, labels, out_dir)

    # Cross-attention heatmaps
    per_sample = pd.read_csv(args.per_sample)
    per_sample = per_sample.merge(
        df_test[["text", "hoax", "root_text", "parent_text"]],
        on="text", how="left",
    )

    for verdict in ["context_wins", "baseline_wins"]:
        subset = per_sample[per_sample["verdict"] == verdict].head(args.n_cases)
        for i, (_, row) in enumerate(subset.iterrows()):
            match = df_test[df_test["text"] == row["text"]].index
            if len(match) == 0:
                continue
            sample_idx = match[0]

            ctx_tokens = ca_tokenizer.convert_ids_to_tokens(all_ctx_ids[sample_idx])
            twt_tokens = ca_tokenizer.convert_ids_to_tokens(all_twt_ids[sample_idx])

            plot_attention_heatmap(
                weights    = all_weights[sample_idx],
                ctx_tokens = ctx_tokens,
                twt_tokens = twt_tokens,
                title      = f"[{verdict}] {row['text'][:70]}...",
                out_path   = out_dir / f"cross_attn_heatmap_{verdict}_{i}.png",
            )

    # Integrated Gradients 
    print(f"\n[IG] Running integrated gradients (n_steps={args.ig_steps}) on case studies...")

    cases = pd.concat([
        per_sample[per_sample["verdict"] == "context_wins"].head(args.n_cases),
        per_sample[per_sample["verdict"] == "baseline_wins"].head(args.n_cases),
    ]).reset_index(drop=True)

    ig_results = run_ig_case_studies(
        baseline_model, baseline_tokenizer,
        hier_model,     hier_tokenizer,
        ca_model,       ca_tokenizer,
        cases, device, n_steps=args.ig_steps,
    )

    plot_ig_comparison(ig_results, out_dir)

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()