from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from config import CONFIG
from data.loader import load_data
from data.preprocessing import filter_contextual_tweets, ids_to_text
from utils.inference_utils import load_hierarchical, load_cross_attention
from utils.inference_utils import enc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",         required=True)
    p.add_argument("--context_path",         required=True)
    p.add_argument("--bt_path",              default=None)
    p.add_argument("--cross_attention_path", required=True)
    p.add_argument("--results_path",         default="figures/")
    p.add_argument("--batch_size",           type=int, default=32)
    p.add_argument("--n_examples",           type=int, default=10)
    p.add_argument("--top_k",                type=int, default=10)
    p.add_argument("--tweet_indices", type=int, nargs="+", default=None)
    return p.parse_args()


POSITIONS   = ["Root tweet", "Parent tweet", "Target tweet"]
SKIP_TOKENS = {"<pad>", "<s>", "</s>", "|", "▁|", "▁", ""}


# Inference (attention-returning variants — distinct from inference.py)
@torch.no_grad()
def run_hierarchical(model, tokenizer, df, batch_size, device):
    all_probs, all_weights = [], []
    texts        = df["text"].tolist()
    parent_texts = df["parent_text"].tolist()
    root_texts   = df["root_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t_enc = enc(tokenizer, texts[i:i+batch_size],        CONFIG["max_len"], device)
        p_enc = enc(tokenizer, parent_texts[i:i+batch_size], CONFIG["max_len"], device)
        r_enc = enc(tokenizer, root_texts[i:i+batch_size],   CONFIG["max_len"], device)

        out = model(
            root_input_ids=        r_enc["input_ids"],
            root_attention_mask=   r_enc["attention_mask"],
            parent_input_ids=      p_enc["input_ids"],
            parent_attention_mask= p_enc["attention_mask"],
            tweet_input_ids=       t_enc["input_ids"],
            tweet_attention_mask=  t_enc["attention_mask"],
            return_attention_weights=True,
        )

        probs = torch.softmax(out.logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_weights.append(out["attention_weights"].cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_weights)


@torch.no_grad()
def run_cross_attention(model, tokenizer, df, batch_size, device):
    all_probs, all_weights, all_ctx_ids = [], [], []
    texts        = df["text"].tolist()
    root_texts   = df["root_text"].tolist()
    parent_texts = df["parent_text"].tolist()

    for i in range(0, len(texts), batch_size):
        t_b = texts[i:i+batch_size]
        r_b = root_texts[i:i+batch_size]
        p_b = parent_texts[i:i+batch_size]

        context = [
            tokenizer.sep_token.join(x for x in [r, p] if x and str(x).strip())
            for r, p in zip(r_b, p_b)
        ]

        ctx_enc = enc(tokenizer, context, CONFIG["max_len_context"], device)
        twt_enc = enc(tokenizer, t_b,     CONFIG["max_len_tweet"],   device)

        out = model(
            context_input_ids=       ctx_enc["input_ids"],
            context_attention_mask=  ctx_enc["attention_mask"],
            tweet_input_ids=         twt_enc["input_ids"],
            tweet_attention_mask=    twt_enc["attention_mask"],
            return_attention_weights=True,
        )

        attn  = out["attention_weights"].cpu().numpy().mean(axis=1).mean(axis=1)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_weights.extend(list(attn))
        all_ctx_ids.extend(ctx_enc["input_ids"].cpu().tolist())

    return np.vstack(all_probs), all_weights, all_ctx_ids


def get_top_tokens(weights, ctx_ids, tokenizer, top_k):
    token_attn = defaultdict(float)
    count      = 0
    for w, ids in zip(weights, ctx_ids):
        for token, attn in zip(tokenizer.convert_ids_to_tokens(ids), w):
            clean = token.replace("▁", "").strip()
            if clean not in SKIP_TOKENS and len(clean) >= 2:
                token_attn[token] += attn
        count += 1
    normalized = {t: v / count for t, v in token_attn.items()}
    return sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_k]


def get_sample_top_tokens(weights, ctx_ids, tokenizer, top_k=5):
    pairs = [
        (token, attn)
        for token, attn in zip(tokenizer.convert_ids_to_tokens(ctx_ids), weights)
        if (clean := token.replace("▁", "").strip()) not in SKIP_TOKENS and len(clean) >= 2
    ]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]


# generating the report 
def _format_example(
    w, row, idx, lbl,
    ca_probs, ca_weights, ca_ctx_ids, ca_tokenizer,
    hier_probs, hier_weights,
    bt_probs, bt_weights,
    top_k,
):
    w(f"  Tweet:   {row['text'][:120]}{'...' if len(row['text']) > 120 else ''}")
    w(f"  Root:    {str(row.get('root_text',   ''))[:80]}")
    w(f"  Parent:  {str(row.get('parent_text', ''))[:80]}")
    w(f"  Label:   {'Stereotype' if lbl == 1 else 'Not Stereotype'}")

    models_to_show = [
        ("[Cross-Attention]", ca_probs, None),
        ("[Hierarchical]",    hier_probs, hier_weights),
    ]
    if bt_probs is not None:
        models_to_show.append(("[BT-Augmented]", bt_probs, bt_weights))
    for tag, probs, extra in models_to_show:
        pred    = probs[idx].argmax()
        correct = "✓ correct" if pred == lbl else "✗ wrong"
        w(f"  {tag:<20} Prediction: {'Stereotype' if pred == 1 else 'Not Stereotype'} "
          f"({correct}) p={probs[idx][pred]:.3f}")
        if tag == "[Cross-Attention]":
            top_ctx = get_sample_top_tokens(ca_weights[idx], ca_ctx_ids[idx], ca_tokenizer, top_k)
            w("  Top context tokens: " + ", ".join(f"{t}({s:.4f})" for t, s in top_ctx))
        else:
            w("  Thread weights: " +
              ", ".join(f"{POSITIONS[i]}={extra[idx, i]:.3f}" for i in range(3)))


def write_report(
    df_test, labels,
    hier_probs, hier_weights,
    bt_probs,   bt_weights,
    ca_probs,   ca_weights, ca_ctx_ids,
    hier_tokenizer, ca_tokenizer,
    n_examples, top_k, out_path,
    specific_indices=None,
):
    lines = []
    def w(line=""):
        lines.append(line)

    # Cross-attention aggregate top tokens
    w("=" * 70)
    w("CROSS-ATTENTION — TOP ATTENDED CONTEXT TOKENS (FULL TEST SET)")
    w("=" * 70)
    for rank, (token, score) in enumerate(get_top_tokens(ca_weights, ca_ctx_ids, ca_tokenizer, top_k), 1):
        w(f"  {rank:>2}. {token:<25} : {score:.4f}")

    for lbl, name in [(0, "NOT STEREOTYPE"), (1, "STEREOTYPE")]:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        w(); w(f"  --- {name} (class {lbl}) ---")
        for rank, (token, score) in enumerate(
            get_top_tokens([ca_weights[i] for i in mask], [ca_ctx_ids[i] for i in mask], ca_tokenizer, top_k), 1
        ):
            w(f"  {rank:>2}. {token:<25} : {score:.4f}")

    # Hierarchical position weights
    position_models = [("HIERARCHICAL", hier_weights)]
    if bt_weights is not None:
        position_models.append(("BT-AUGMENTED", bt_weights))
    for model_name, weights in position_models:
        w(); w("=" * 70)
        w(f"{model_name} — MEAN THREAD POSITION ATTENTION WEIGHTS")
        w("=" * 70)
        for lbl, name in [(0, "Not stereotype"), (1, "Stereotype")]:
            mask = labels == lbl
            w(f"  {name}:")
            for i, pos in enumerate(POSITIONS):
                w(f"    {pos:<20} : {weights[mask, i].mean():.4f}")

    # Per-example details
    for lbl, class_name in [(0, "NOT STEREOTYPE"), (1, "STEREOTYPE")]:
        w(); w("=" * 70); w(f"EXAMPLE DETAILS — {class_name}"); w("=" * 70)
        for ex_num, idx in enumerate(
            [i for i, l in enumerate(labels) if l == lbl][:n_examples], 1
        ):
            w(); w(f"--- Example {ex_num} (idx {idx}) ---")
            _format_example(w, df_test.iloc[idx], idx, lbl,
                            ca_probs, ca_weights, ca_ctx_ids, ca_tokenizer,
                            hier_probs, hier_weights, bt_probs, bt_weights, top_k)

    # Specific indices
    if specific_indices:
        w(); w("=" * 70); w("EXPLICITLY REQUESTED TWEET INDICES"); w("=" * 70)
        for ex_num, idx in enumerate(specific_indices, 1):
            w()
            if idx < 0 or idx >= len(df_test):
                w(f"--- Index {idx} out of range (0..{len(df_test)-1}) ---")
                continue
            w(f"--- Requested Example {ex_num} (idx {idx}) ---")
            _format_example(w, df_test.iloc[idx], idx, labels[idx],
                            ca_probs, ca_weights, ca_ctx_ids, ca_tokenizer,
                            hier_probs, hier_weights, bt_probs, bt_weights, top_k)

    return "\n".join(lines)


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
    print(f"[data] {len(df_test)} test samples")

    specific_indices = args.tweet_indices or []

    print("\n[loading] Hierarchical...")
    hier_model, hier_tokenizer = load_hierarchical(args.context_path, device)
    hier_probs, hier_weights   = run_hierarchical(hier_model, hier_tokenizer, df_test, args.batch_size, device)

    print("\n[loading] BT-Augmented...")
    bt_model, bt_tokenizer = load_hierarchical(args.bt_path, device)
    bt_probs, bt_weights   = run_hierarchical(bt_model, bt_tokenizer, df_test, args.batch_size, device)

    print("\n[loading] Cross-Attention...")
    ca_model, ca_tokenizer             = load_cross_attention(args.cross_attention_path, device)
    ca_probs, ca_weights, ca_ctx_ids   = run_cross_attention(ca_model, ca_tokenizer, df_test, args.batch_size, device)

    print("\n[report] Generating...")
    report = write_report(
        df_test, labels,
        hier_probs, hier_weights,
        bt_probs,   bt_weights,
        ca_probs,   ca_weights, ca_ctx_ids,
        hier_tokenizer, ca_tokenizer,
        n_examples=args.n_examples,
        top_k=args.top_k,
        out_path=out_dir,
        specific_indices=specific_indices,
    )

    print("\n" + report)
    out = out_dir / "attention_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()