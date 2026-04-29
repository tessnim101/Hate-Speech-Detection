"""
Model A — XLM-R-base embeddings (text only, CLS token)
=======================================================
Output: embeddings_model_a.pt
  {
    "embeddings":      Tensor [N, 768]   float32  — CLS vectors
    "stereotype":      Tensor [N]        int64    — hard label
    "stereotype_soft": Tensor [N]        float32  — soft label
    "stereotype_a1":   Tensor [N]        int64    — annotator 1
    "stereotype_a2":   Tensor [N]        int64    — annotator 2
    "stereotype_a3":   Tensor [N]        int64    — annotator 3
    "implicit":        Tensor [N]        int64    — hard label
    "implicit_soft":   Tensor [N]        float32  — soft label
    "implicit_a1":     Tensor [N]        int64    — annotator 1
    "implicit_a2":     Tensor [N]        int64    — annotator 2
    "implicit_a3":     Tensor [N]        int64    — annotator 3
    "id":              list[str]                  — row id
    "comment_id":      list[str]                  — comment id
    "source":          list[str]                  — "detests" | "stereohoax"
    "text":            list[str]                  — raw sentence
    "level1":          list[str|None]             — previous sentence id
    "level2":          list[str|None]             — previous comment id
    "level3":          list[str|None]             — thread head comment id
    "level4":          list[str|None]             — news/hoax id
  }

Usage
-----
    python embed_model_a.py
    python embed_model_a.py --batch-size 64 --max-length 128 --split train
    python embed_model_a.py --split test   # held-out split (no labels)

Requirements
------------
    pip install torch transformers datasets tqdm
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="xlm-roberta-base")
    p.add_argument("--dataset",    default="CLiC-UB/DETESTS-Dis")
    p.add_argument("--split",      default="train")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output",     default="embeddings_model_a.pt")
    p.add_argument("--fp16",       action="store_true",
                   help="Use half precision on CUDA (faster, ~same quality)")
    return p.parse_args()


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        dev  = torch.device("mps")
        print("GPU: Apple MPS")
    else:
        dev  = torch.device("cpu")
        print("No GPU found — running on CPU (will be slow)")
    return dev


# ── column specs ──────────────────────────────────────────────────────────────

# Columns saved as tensors (absent in the test split → skipped gracefully)
TENSOR_COLS = {
    "stereotype":      torch.int64,
    "stereotype_soft": torch.float32,
    "stereotype_a1":   torch.int64,
    "stereotype_a2":   torch.int64,
    "stereotype_a3":   torch.int64,
    "implicit":        torch.int64,
    "implicit_soft":   torch.float32,
    "implicit_a1":     torch.int64,
    "implicit_a2":     torch.int64,
    "implicit_a3":     torch.int64,
}

# Columns kept as plain Python lists (strings / None)
LIST_COLS = ["id", "comment_id", "source", "text",
             "level1", "level2", "level3", "level4"]


# ── collate ───────────────────────────────────────────────────────────────────

def make_collate(tokenizer, max_length):
    def collate(batch):
        # tokenise the sentence text for XLM-R
        enc = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # string / None columns — collect as lists
        meta = {col: [item.get(col) for item in batch] for col in LIST_COLS}

        # numeric columns → tensors (skipped if absent, e.g. test split)
        for col, dtype in TENSOR_COLS.items():
            if col in batch[0] and batch[0][col] is not None:
                meta[col] = torch.tensor([item[col] for item in batch], dtype=dtype)

        return enc, meta
    return collate


# ── embed ─────────────────────────────────────────────────────────────────────

def embed(args, device):
    print(f"\nLoading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModel.from_pretrained(args.model)
    model.eval().to(device)

    if args.fp16 and device.type == "cuda":
        model = model.half()
        print("Running in fp16")

    print(f"\nLoading dataset: {args.dataset} / {args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  {len(ds):,} rows")

    collate = make_collate(tokenizer, args.max_length)
    loader  = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # accumulators
    all_embeddings = []
    list_buckets   = {col: [] for col in LIST_COLS}
    tensor_buckets = {col: [] for col in TENSOR_COLS}

    print(f"\nEncoding — batch_size={args.batch_size}, max_length={args.max_length}")
    with torch.no_grad():
        for enc, meta in tqdm(loader, unit="batch"):
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]  # [B, 768]  CLS token
            all_embeddings.append(cls.cpu().float())

            for col in LIST_COLS:
                list_buckets[col].extend(meta[col])

            for col in TENSOR_COLS:
                if col in meta:
                    tensor_buckets[col].append(meta[col])

    # assemble final payload
    embeddings = torch.cat(all_embeddings, dim=0)  # [N, 768]
    print(f"\nEmbedding matrix: {embeddings.shape}  "
          f"({embeddings.numel() * 4 / 1e6:.1f} MB)")

    payload = {"embeddings": embeddings}

    for col in LIST_COLS:
        payload[col] = list_buckets[col]

    for col in TENSOR_COLS:
        if tensor_buckets[col]:
            payload[col] = torch.cat(tensor_buckets[col], dim=0)

    torch.save(payload, args.output)
    print(f"Saved → {args.output}")
    return payload


# ── sanity check ──────────────────────────────────────────────────────────────

def sanity_check(payload):
    emb = payload["embeddings"]
    print("\n── sanity check ──────────────────────────────")
    print(f"  shape:           {emb.shape}")
    print(f"  dtype:           {emb.dtype}")
    print(f"  mean norm:       {emb.norm(dim=1).mean():.3f}")
    print(f"  first row[:5]:   {emb[0, :5].tolist()}")
    if "stereotype" in payload:
        labels = payload["stereotype"]
        pos    = labels.sum().item()
        total  = len(labels)
        print(f"  stereotype=1:    {pos:,} / {total:,} ({100*pos/total:.1f}%)")
    saved_keys = [k for k in payload if k != "embeddings"]
    print(f"  extra keys:      {saved_keys}")
    print("──────────────────────────────────────────────")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args    = parse_args()
    device  = get_device()
    payload = embed(args, device)
    sanity_check(payload)
    print("\nDone. Load with:")
    print(f'  data = torch.load("{args.output}")')
    print('  X    = data["embeddings"]      # [N, 768]  features')
    print('  y    = data["stereotype"]      # [N]       hard label')
    print('  ids  = data["id"]              # list[str] row ids')
    print('  ctx  = data["level1"]          # list[str] context for Model B')
