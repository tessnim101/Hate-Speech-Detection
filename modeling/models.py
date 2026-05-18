"""
Load pre-trained model and define custom models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification , AutoTokenizer


def load_model(model_name: str, num_labels: int = 2):
    """
    Load pre-trained model.
    """
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
            num_heads=8,
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
        """
        Return [CLS] token representation for a batch of texts.
        """
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
        root_cls   = self.encode(root_input_ids,   root_attention_mask) # (B, H)
        parent_cls = self.encode(parent_input_ids, parent_attention_mask)  
        tweet_cls  = self.encode(tweet_input_ids,  tweet_attention_mask) 

        # Stack into a sequence of 3 "thread tokens": (B, 3, H)
        positions = torch.arange(3, device=root_cls.device)
        pos_emb = self.position_embeddings(positions).unsqueeze(0)  # (1, 3, H)

        thread = torch.stack([root_cls, parent_cls, tweet_cls], dim=1) + pos_emb

        # Sequences with only special tokens ([CLS] + [SEP]) have attention_mask sum == 2
        root_empty   = (root_attention_mask.sum(dim=1)   <= 2)
        parent_empty = (parent_attention_mask.sum(dim=1) <= 2)
        tweet_empty  = torch.zeros(root_empty.shape[0], dtype=torch.bool, device=root_cls.device)
        key_padding_mask = torch.stack([root_empty, parent_empty, tweet_empty], dim=1)

        attn_out, _ = self.thread_attention(
            query=thread[:, 2:, :], # tweet CLS as query  (B, 1, H)
            key=thread, # full thread as keys  (B, 3, H)
            value=thread, # full thread as vals  (B, 3, H)
            key_padding_mask=key_padding_mask,
        )
        # attn_out: (B, 1, H), tweet representation enriched by thread context
        pooled = attn_out.squeeze(1)  # (B, H)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        # Return a dict for HuggingFace Trainer compatibility.
        return {"logits": logits}


class CrossAttentionHoaxModel(nn.Module):
    """
    Token-level cross-attention model for racial hoax detection.

    Unlike the CLS-level hierarchical model, this model allows every
    tweet token to attend to every context token (hoax + root + parent),
    capturing fine-grained interactions between the tweet and its context.

    Args:
        model_name:  HuggingFace model name or local path
        num_labels:  number of output classes (default: 2)
        hidden_dim:  intermediate projection size (default: 256)
        num_heads:   number of attention heads (default: 8)
        dropout:     dropout rate (default: 0.1)
    """

    def __init__(
        self,
        model_name:  str,
        num_labels:  int   = 2,
        hidden_dim:  int   = 256,
        num_heads:   int   = 8,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder   = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        hidden = self.encoder.config.hidden_size  # 768

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return all token embeddings for a batch of texts."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state  # (B, seq_len, 768)

    def forward(
        self,
        context_input_ids:      torch.Tensor,
        context_attention_mask: torch.Tensor,
        tweet_input_ids:        torch.Tensor,
        tweet_attention_mask:   torch.Tensor,
        labels:                 torch.Tensor | None = None,
        class_weights:          torch.Tensor | None = None,
        return_attention_weights: bool = False
    ) -> dict:
        """
        Args:
            context_input_ids:      (B, ctx_len)   hoax + root + parent concatenated
            context_attention_mask: (B, ctx_len)
            tweet_input_ids:        (B, tweet_len)
            tweet_attention_mask:   (B, tweet_len)
            labels:                 (B,) long tensor, optional
            class_weights:          (num_labels,) float tensor, optional
            return_attention_weights: bool, optional
        Returns:
            dict with 'logits' (B, num_labels) and optionally 'loss'
        """
        context_tokens = self.encode(context_input_ids, context_attention_mask)  # (B, ctx_len, 768)
        tweet_tokens   = self.encode(tweet_input_ids,   tweet_attention_mask)    # (B, tweet_len, 768)

        # True where tokens should be ignored by attention
        context_key_padding_mask = (context_attention_mask == 0)  # (B, ctx_len)

        # Cross-attention: tweet tokens attend to context tokens
        attended, attn_weights = self.cross_attention(
        query=tweet_tokens,
        key=context_tokens,
        value=context_tokens,
        key_padding_mask=context_key_padding_mask,
        need_weights=True,        # ← always compute
        average_attn_weights=False,  # ← keep per-head weights
            )
        
        # Residual connection + layer norm
        enriched = self.layer_norm(tweet_tokens + attended)  # (B, tweet_len, 768)

        # [CLS] token for classification
        cls_repr = enriched[:, 0, :]   # (B, 768)
        logits   = self.classifier(cls_repr)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)

        out = {"logits": logits, "loss": loss}
        if return_attention_weights:
            out["attention_weights"] = attn_weights  # (B, num_heads, tweet_len, ctx_len)
        return out

    @torch.no_grad()
    def predict_proba(
        self,
        context_input_ids:      torch.Tensor,
        context_attention_mask: torch.Tensor,
        tweet_input_ids:        torch.Tensor,
        tweet_attention_mask:   torch.Tensor,
    ) -> torch.Tensor:
        """Returns softmax probabilities (B, num_labels) for inference."""
        out = self.forward(
            context_input_ids, context_attention_mask,
            tweet_input_ids,   tweet_attention_mask,
        )
        return torch.softmax(out["logits"], dim=-1)