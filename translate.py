"""
GPU back-translation using Helsinki-NLP/opus-mt MarianMT models.
Translates text, hoax, parent_text, and root_text to fix the context
mismatch in the original augmentation (all fields back-translated together).

On RCP:
    runai submit bt-translate \
      --run-as-uid 244835 \
      --image registry.rcp.epfl.ch/ee-559-bechrifa/my-toolbox:v0.7 \
      --project course-ee-559-bechrifa \
      --gpu 1 \
      --existing-pvc claimname=home,path=/home/bechrifa \
      -e HF_TOKEN=<token> \
      --command -- python3 /home/bechrifa/Hate-Speech-Detection/translate.py \
        --data_dir /home/bechrifa/Hate-Speech-Detection/data/spanish_subset/
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from data.loader import load_data
from data.preprocessing import ids_to_text

COLUMNS = ["hoax"]


def load_model(src: str, tgt: str, device: str):
    name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    print(f"  Loading {name}...")
    tok   = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to(device)
    model.eval()
    return tok, model


def translate_batch(texts: list[str], tok, model, device: str, max_len: int) -> list[str]:
    inputs = tok(texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_len)
    return tok.batch_decode(out, skip_special_tokens=True)


def back_translate(texts: list[str], es_en, en_es, device: str,
                   batch_size: int, max_len: int) -> list[str]:
    """Deduplicated ES→EN→ES back-translation."""
    es_en_tok, es_en_model = es_en
    en_es_tok, en_es_model = en_es

    unique = [t for t in dict.fromkeys(texts) if isinstance(t, str) and t.strip()]
    print(f"    {len(unique)} unique texts (from {len(texts)} total)")

    en_map, es_map = {}, {}

    for i in tqdm(range(0, len(unique), batch_size), desc="ES→EN"):
        batch = unique[i:i + batch_size]
        translated = translate_batch(batch, es_en_tok, es_en_model, device, max_len)
        en_map.update(zip(batch, translated))

    intermediates = list(en_map.values())
    unique_en = list(dict.fromkeys(intermediates))
    for i in tqdm(range(0, len(unique_en), batch_size), desc="EN→ES"):
        batch = unique_en[i:i + batch_size]
        translated = translate_batch(batch, en_es_tok, en_es_model, device, max_len)
        es_map.update(zip(batch, translated))

    lookup = {orig: es_map[en_map[orig]] for orig in unique}
    return [lookup.get(t, t) if isinstance(t, str) and t.strip() else "" for t in texts]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/spanish_subset/")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len",    type=int, default=128)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[device] {args.device}")

    df_train, _ = load_data(args.data_dir)
    print(f"Loaded {len(df_train)} training rows")

    # Derive bt_text, bt_parent_text, bt_root_text from existing train_bt.csv
    # (parent and root are tweets already in the dataset, so we reuse their translations)
    bt_path = Path(args.data_dir) / "train_bt.csv"
    if not bt_path.exists():
        raise FileNotFoundError(f"Existing back-translation file not found: {bt_path}")
    bt_existing = pd.read_csv(bt_path).rename(columns={"backtranslated_text": "bt_text"})
    bt_lookup = bt_existing.set_index("comment_id")["bt_text"].to_dict()

    out = pd.DataFrame({"comment_id": df_train["comment_id"]})
    out["bt_text"]        = df_train["comment_id"].map(bt_lookup).fillna("")
    out["bt_parent_text"] = df_train["level2"].map(bt_lookup).fillna("")
    out["bt_root_text"]   = df_train["level3"].map(bt_lookup).fillna("")
    print(f"[reused] bt_text / bt_parent_text / bt_root_text from {bt_path}")

    print("\nLoading translation models...")
    es_en = load_model("es", "en", args.device)
    en_es = load_model("en", "es", args.device)

    for col in COLUMNS:
        if col not in df_train.columns:
            print(f"Skipping {col} (not found)")
            continue
        print(f"\nBack-translating: {col}")
        out[f"bt_{col}"] = back_translate(
            df_train[col].fillna("").tolist(),
            es_en, en_es, args.device, args.batch_size, args.max_len,
        )

    out.to_csv(bt_path, index=False)
    print(f"\n[saved] {bt_path}  ({len(out)} rows, cols: {list(out.columns)})")


if __name__ == "__main__":
    main()
