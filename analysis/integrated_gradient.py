"""
Integrated Gradients interpretability analysis.

Runs IG on tweet tokens across baseline, hierarchical, and cross-attention
models on case studies drawn from inference_per_sample.csv.

python analysis/integrated_gradient.py \
    --dataset_path          "data/spanish_subset_collapsed/" \
    --baseline_path         "results/best_model_baseline/" \
    --context_path          "results/best_model_hierarchical/" \
    --cross_attention_path  "results/best_model_cross_attention/" \
    --results_path          "figures/" \
    --per_sample            "results/inference_per_sample.csv" \
    --tweet_idx             599
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
SPANISH_STOPWORDS = set(nltk.corpus.stopwords.words("spanish"))
from captum.attr import IntegratedGradients

from config import CONFIG
from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text, clean_df
from utils.inference_utils import enc, load_hierarchical, load_cross_attention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",         required=True)
    p.add_argument("--baseline_path",        required=True)
    p.add_argument("--context_path",         required=True)
    p.add_argument("--cross_attention_path", required=True)
    p.add_argument("--per_sample",           required=True)
    p.add_argument("--results_path",         required=True)
    p.add_argument("--batch_size",           type=int, default=32)
    p.add_argument("--n_cases",              type=int, default=4)
    p.add_argument("--ig_steps",             type=int, default=50)
    p.add_argument("--tweet_idx",            type=int, default=None)
    return p.parse_args()


def load_baseline(path, device):
    model     = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


class BaselineIGWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_embeds, attention_mask):
        out = self.model.roberta(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        return self.model.classifier(out.last_hidden_state)


class HierarchicalIGWrapper(nn.Module):
    """
    IG on tweet embeddings; root and parent are held fixed.
    Matches HierarchicalContextModel.forward exactly:
      - encode uses CLS token (last_hidden_state[:, 0, :])
      - tweet queries the full thread via thread_attention
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        tweet_embeds,
        tweet_mask,
        root_ids,
        root_mask,
        parent_ids,
        parent_mask,
    ):
        # Encode root and parent via CLS (matches model.encode)
        root_repr   = self.model.encoder(
            input_ids=root_ids, attention_mask=root_mask
        ).last_hidden_state[:, 0, :]  # (B, H)

        parent_repr = self.model.encoder(
            input_ids=parent_ids, attention_mask=parent_mask
        ).last_hidden_state[:, 0, :]  # (B, H)

        # Encode tweet via embeddings (for IG) then CLS
        tweet_out  = self.model.encoder(
            inputs_embeds=tweet_embeds, attention_mask=tweet_mask
        )
        tweet_repr = tweet_out.last_hidden_state[:, 0, :]  # (B, H) — CLS

        # Thread assembly with position embeddings
        positions = torch.arange(3, device=tweet_repr.device)
        pos_emb   = self.model.position_embeddings(positions).unsqueeze(0)  # (1, 3, H)
        thread    = torch.stack([root_repr, parent_repr, tweet_repr], dim=1) + pos_emb

        # Padding mask
        root_empty   = (root_mask.sum(dim=1)   <= 2)
        parent_empty = (parent_mask.sum(dim=1) <= 2)
        tweet_empty  = torch.zeros(root_empty.shape[0], dtype=torch.bool,
                                   device=tweet_repr.device)
        key_padding_mask = torch.stack([root_empty, parent_empty, tweet_empty], dim=1)

        # Tweet queries the full thread
        attn_out, _ = self.model.thread_attention(
            query=thread[:, 2:, :],
            key=thread, value=thread,
            key_padding_mask=key_padding_mask,
        )

        tweet_hidden = self.model.norm1(tweet_repr + attn_out.squeeze(1))
        tweet_hidden = self.model.norm2(tweet_hidden + self.model.ffn(tweet_hidden))
        return self.model.classifier(tweet_hidden)


class CrossAttentionIGWrapper(nn.Module):
    """
    IG on tweet embeddings; context IDs are held fixed.
    Matches CrossAttentionContextModel.forward exactly:
      - pooling uses CLS token (enriched[:, 0, :])
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tweet_embeds, tweet_mask, ctx_ids, ctx_mask):
        context_tokens       = self.model.encode(ctx_ids, ctx_mask)
        ctx_key_padding_mask = (ctx_mask == 0)

        tweet_out = self.model.encoder(
            inputs_embeds=tweet_embeds,
            attention_mask=tweet_mask,
        )
        enriched = tweet_out.last_hidden_state

        for layer in self.model.cross_layers:
            attended, _ = layer["cross_attn"](
                query=enriched,
                key=context_tokens,
                value=context_tokens,
                key_padding_mask=ctx_key_padding_mask,
            )
            enriched = layer["norm1"](enriched + attended)
            enriched = layer["norm2"](enriched + layer["ffn"](enriched))

        # CLS token — matches model.forward
        pooled = enriched[:, 0, :]  # (B, H)
        return self.model.classifier(pooled)



def get_embeddings(model, model_type, input_ids):
    """Extract word embeddings from the model's embedding layer."""
    with torch.no_grad():
        if model_type == "baseline":
            return model.roberta.embeddings.word_embeddings(input_ids)
        else:
            return model.encoder.embeddings.word_embeddings(input_ids)


def compute_ig(wrapper, input_embeds, additional_args, target_class, n_steps):
    """
    Run Integrated Gradients on input_embeds.
    Returns (seq_len,) array of per-token importance (summed across embed dim).
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


# Case study IG 

def run_ig_case_studies(
    baseline_model,  baseline_tokenizer,
    hier_model,      hier_tokenizer,
    ca_model,        ca_tokenizer,
    cases_df, device, n_steps,
):
    b_wrapper  = BaselineIGWrapper(baseline_model).to(device).eval()
    h_wrapper  = HierarchicalIGWrapper(hier_model).to(device).eval()
    ca_wrapper = CrossAttentionIGWrapper(ca_model).to(device).eval()

    results = []

    for _, row in cases_df.iterrows():
        tweet   = row["text"]
        label   = int(row["stereotype"])
        verdict = row.get("verdict", "")

        # Baseline 
        b_enc    = enc(baseline_tokenizer, [tweet], CONFIG["max_len"], device)
        b_embeds = get_embeddings(baseline_model, "baseline", b_enc["input_ids"])
        b_embeds = b_embeds.detach().requires_grad_(True)
        b_imp    = compute_ig(
            b_wrapper, b_embeds,
            additional_args=(b_enc["attention_mask"],),
            target_class=label, n_steps=n_steps,
        )
        b_tokens = baseline_tokenizer.convert_ids_to_tokens(b_enc["input_ids"][0].tolist())

        # Hierarchical 
        h_enc      = enc(hier_tokenizer, [tweet],                      CONFIG["max_len"], device)
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

        # Cross-Attention 
        context = hier_tokenizer.sep_token.join(
            x for x in [row.get("root_text", ""), row.get("parent_text", "")]
            if x and str(x).strip()
        )
        ca_twt_enc = enc(ca_tokenizer, [tweet],   CONFIG["max_len_tweet"],   device)
        ca_ctx_enc = enc(ca_tokenizer, [context], CONFIG["max_len_context"], device)

        ca_embeds = get_embeddings(ca_model, "cross_attention", ca_twt_enc["input_ids"])
        ca_embeds = ca_embeds.detach().requires_grad_(True)
        ca_imp    = compute_ig(
            ca_wrapper, ca_embeds,
            additional_args=(
                ca_twt_enc["attention_mask"],
                ca_ctx_enc["input_ids"],
                ca_ctx_enc["attention_mask"],
            ),
            target_class=label, n_steps=n_steps,
        )
        ca_tokens = ca_tokenizer.convert_ids_to_tokens(ca_twt_enc["input_ids"][0].tolist())

        results.append({
            "tweet":           tweet,
            "label":           label,
            "verdict":         verdict,
            "root_text":       row.get("root_text",   ""),
            "parent_text":     row.get("parent_text", ""),
            "baseline":        {"tokens": b_tokens,  "importance": b_imp},
            "hierarchical":    {"tokens": h_tokens,  "importance": h_imp},
            "cross_attention": {"tokens": ca_tokens, "importance": ca_imp},
        })

    return results


def plot_ig_comparison(ig_results, out_dir):
    model_keys   = ["baseline", "hierarchical", "cross_attention"]
    model_labels = ["Baseline", "Hierarchical", "Cross-Attention"]
    model_colors = ["#6C8EBF", "#D4763B", "#55A868"]
    skip         = {"<s>", "</s>", "<pad>"}
    top_k        = 20

    for idx, result in enumerate(ig_results):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))

        root   = result.get("root_text",   "N/A")[:80]
        parent = result.get("parent_text", "N/A")[:80]

        fig.suptitle(
            f"[{result['verdict']}]  label={result['label']}  |  "
            f"{result['tweet'][:80]}{'...' if len(result['tweet']) > 80 else ''}\n\n"
            f"Root:   {root}\nParent: {parent}",
            fontsize=8, fontweight="bold", ha="left", x=0.01,
        )
        plt.subplots_adjust(top=0.72)

        for ax, key, mlabel, color in zip(axes, model_keys, model_labels, model_colors):
            tokens     = result[key]["tokens"]
            importance = result[key]["importance"]

            filtered = [
                (t, s) for t, s in zip(tokens, importance)
                if t not in skip
                and t.startswith("▁")
                and len(t.replace("▁", "").strip()) >= 3
                and t.replace("▁", "").strip().lower() not in SPANISH_STOPWORDS
                and not t.replace("▁", "").strip().isdigit()
            ]

            if not filtered:
                ax.set_title(f"{mlabel}\n(no tokens after filtering)", fontsize=10)
                continue

            tokens_f, scores_f = zip(*filtered)
            scores_f = np.array(scores_f)

            keep        = np.argsort(np.abs(scores_f))[-top_k:][::-1]
            tokens_f    = [tokens_f[i] for i in keep]
            scores_f    = scores_f[keep]
            scores_norm = scores_f / (np.abs(scores_f).max() + 1e-9)

            bar_colors = [color if s >= 0 else "#e74c3c" for s in scores_norm]
            y_pos      = np.arange(len(tokens_f))

            ax.barh(y_pos, scores_norm, color=bar_colors, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens_f, fontsize=9)
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

    # Data loading — must match training preprocessing
    df_train, df_test = load_data(args.dataset_path)
    df_train = clean_df(df_train)
    df_test  = clean_df(df_test)
    combined = pd.concat([df_train, df_test], ignore_index=True)
    df_train = ids_to_text(df_train, lookup_df=combined)
    df_test  = ids_to_text(df_test,  lookup_df=combined)
    df_test  = filter_contextual_tweets(df_test)

    print("[loading] Baseline...")
    baseline_model, baseline_tokenizer = load_baseline(args.baseline_path, device)
    print("[loading] Hierarchical...")
    hier_model, hier_tokenizer = load_hierarchical(args.context_path, device)
    print("[loading] Cross-Attention...")
    ca_model, ca_tokenizer = load_cross_attention(args.cross_attention_path, device)

    if args.tweet_idx is not None:
        row   = df_test.iloc[args.tweet_idx]
        cases = pd.DataFrame([{
            "text":        row["text"],
            "stereotype":  row["stereotype"],
            "verdict":     "manual",
            "root_text":   row.get("root_text",   ""),
            "parent_text": row.get("parent_text", ""),
        }])
    else:
        per_sample = pd.read_csv(args.per_sample)
        per_sample = per_sample.merge(
            df_test[["text", "root_text", "parent_text"]],
            on="text", how="left",
        )
        cases = pd.concat([
            per_sample[per_sample["verdict"] == "context_wins"].head(args.n_cases),
            per_sample[per_sample["verdict"] == "baseline_wins"].head(args.n_cases),
        ]).reset_index(drop=True)

    print(f"\n[IG] Running on {len(cases)} cases (n_steps={args.ig_steps})...")
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