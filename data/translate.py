import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

TEXT_COL, ID_COL = "text", "comment_id"
PARENT_COL, ROOT_COL = "level2", "level3"


def load_model(src: str, tgt: str, device: str):
    name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    print(f"  Loading {name}...")
    tok   = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to(device)
    model.eval()
    return tok, model


def translate_batch(texts, tok, model, device, max_len):
    inputs = tok(texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_len)
    return tok.batch_decode(out, skip_special_tokens=True)


def back_translate(texts, es_en, en_es, device, batch_size, max_len):
    """Deduplicated ES->EN->ES back-translation. Preserves input order/length."""
    es_en_tok, es_en_model = es_en
    en_es_tok, en_es_model = en_es

    unique = [t for t in dict.fromkeys(texts) if isinstance(t, str) and t.strip()]
    print(f"    {len(unique)} unique texts (from {len(texts)} total)")

    en_map = {}
    for i in tqdm(range(0, len(unique), batch_size), desc="ES->EN"):
        batch = unique[i:i + batch_size]
        en_map.update(zip(batch, translate_batch(batch, es_en_tok, es_en_model, device, max_len)))

    unique_en = list(dict.fromkeys(en_map.values()))
    es_map = {}
    for i in tqdm(range(0, len(unique_en), batch_size), desc="EN->ES"):
        batch = unique_en[i:i + batch_size]
        es_map.update(zip(batch, translate_batch(batch, en_es_tok, en_es_model, device, max_len)))

    lookup = {orig: es_map[en_map[orig]] for orig in unique}
    return [lookup.get(t, t) if isinstance(t, str) and t.strip() else "" for t in texts]


def remap_id(x: str) -> str:
    """Point a twin's parent/root at the parent/root twin; keep the '0' sentinel."""
    x = str(x)
    return x if x == "0" else f"{x}_bt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/spanish_subset_collapsed/")
    p.add_argument("--out_dir",    default="data/spanish_subset_collapsed_aug/")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len",    type=int, default=128)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[device] {args.device}")

    data_dir = Path(args.data_dir)
    # Read exactly like main.py does (raw), so original rows stay byte-identical.
    raw_train = pd.read_csv(data_dir / "train.csv")
    raw_test  = pd.read_csv(data_dir / "test.csv")
    print(f"Loaded {len(raw_train)} train / {len(raw_test)} test rows")

    # One translation pass over the full corpus, keyed by comment_id.
    corpus = (pd.concat([raw_train, raw_test], ignore_index=True)
                .drop_duplicates(subset=ID_COL, keep="first").reset_index(drop=True))
    print(f"Corpus: {len(corpus)} unique comments")

    print("\nLoading translation models...")
    es_en = load_model("es", "en", args.device)
    en_es = load_model("en", "es", args.device)

    print(f"\nBack-translating: {TEXT_COL}")
    bt = back_translate(corpus[TEXT_COL].fillna("").tolist(),
                        es_en, en_es, args.device, args.batch_size, args.max_len)
    bt_map = dict(zip(corpus[ID_COL].astype(str), bt))

    # Build twins from TRAIN comments only.
    twins = raw_train.copy()
    twins[ID_COL] = twins[ID_COL].astype(str)
    twins[TEXT_COL] = twins[ID_COL].map(bt_map).fillna("")
    twins = twins[twins[TEXT_COL].str.strip() != ""].copy()
    twins[PARENT_COL] = twins[PARENT_COL].astype(str).map(remap_id)
    twins[ROOT_COL]   = twins[ROOT_COL].astype(str).map(remap_id)
    twins[ID_COL]     = twins[ID_COL] + "_bt"

    # Originals as strings so ids line up with twin ids after concat.
    base = raw_train.copy()
    for c in (ID_COL, PARENT_COL, ROOT_COL):
        base[c] = base[c].astype(str)

    aug = pd.concat([base, twins], ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out_dir / "train.csv", index=False)
    shutil.copy(data_dir / "test.csv", out_dir / "test.csv")  # test unchanged

    print(f"\n[saved] {out_dir/'train.csv'}  ({len(base)} originals + {len(twins)} twins = {len(aug)})")
    print(f"[saved] {out_dir/'test.csv'}  (copied unchanged)")
    print(f"\nRun:  python main.py --dataset_path {out_dir}/ --results_path results/ "
          f"--imbalance_strategy class_weights")


if __name__ == "__main__":
    main()
