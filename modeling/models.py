"""
Load model
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification


def load_model(model_name: str, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


class HierarchicalContextModel(nn.Module):
    """
    Encodes root, parent, and tweet separately using a shared XLM-RoBERTa encoder,
    then combines their [CLS] representations via multi-head attention before classification.
    """

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768 for XLM-RoBERTa base

        self.thread_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.position_embeddings = nn.Embedding(3, hidden)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels),
        )

    def encode(self, input_ids, attention_mask):
        """Return [CLS] token representation for a batch of texts."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]

    def forward(
        self,
        root_input_ids,      
        root_attention_mask,
        parent_input_ids,    
        parent_attention_mask,
        tweet_input_ids,     
        tweet_attention_mask,
        labels=None,
    ):
        root_cls   = self.encode(root_input_ids,   root_attention_mask)    # (B, H)
        parent_cls = self.encode(parent_input_ids, parent_attention_mask)  # (B, H)
        tweet_cls  = self.encode(tweet_input_ids,  tweet_attention_mask)   # (B, H)

        # Stack into a sequence of 3 "thread tokens": (B, 3, H)
        positions = torch.arange(3, device=root_cls.device)
        pos_emb = self.position_embeddings(positions).unsqueeze(0)  # (1, 3, H)

        thread = torch.stack([root_cls, parent_cls, tweet_cls], dim=1) + pos_emb

        # Tweet (last position) attends to the full thread
        attn_out, _ = self.thread_attention(
            query=thread[:, 2:, :],   # tweet CLS as query  (B, 1, H)
            key=thread,               # full thread as keys  (B, 3, H)
            value=thread,             # full thread as vals  (B, 3, H)
        )
        # attn_out: (B, 1, H) — tweet representation enriched by thread context
        pooled = attn_out.squeeze(1)  # (B, H)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        # Return a dict so HuggingFace Trainer works out of the box
        return {"logits": logits}
