from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from config import CONFIG
from modeling.models import CrossAttentionContextModel, HierarchicalContextModel

def enc(tokenizer, texts, max_len, device):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

def load_hierarchical(path, device):
    model   = HierarchicalContextModel(CONFIG["model_name"]).to(device)
    weights = load_file(str(Path(path) / "model.safetensors"), device=str(device))
    model.load_state_dict(weights)
    model.eval()
    return model, AutoTokenizer.from_pretrained(path)


def load_cross_attention(path, device):
    model   = CrossAttentionContextModel(CONFIG["model_name"]).to(device)
    weights = load_file(str(Path(path) / "model.safetensors"), device=str(device))
    model.load_state_dict(weights)
    model.eval()
    return model, AutoTokenizer.from_pretrained(path)