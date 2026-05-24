import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from typing import Optional


def load_model(model_name: str, num_labels: int = 2):
    """Load pre-trained model for fine-tuning."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings over non-padding positions.
    More robust than CLS-only when sequences vary in length.
    """
    mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
    summed = (hidden_states * mask).sum(dim=1)           # (B, H)
    lengths = mask.sum(dim=1).clamp(min=1e-9)            # (B, 1)
    return summed / lengths                              # (B, H)


class HierarchicalContextModel(nn.Module):
    """
    Encodes root, parent, and tweet separately using a shared encoder.
    The tweet CLS queries the root and parent representations to retrieve
    relevant context, then is classified.

    Context levels (coarse → fine):
        root   – first comment/tweet in thread  (level3)
        parent – previous comment/tweet          (level2)
        tweet  – the target sentence/tweet       (text)

    The tweet is the query: it asks "what in the context is relevant
    to classifying me?" Root and parent are keys/values — passive context
    that gets retrieved. This is the correct information flow for the
    task of context-aided tweet classification.
    """

    def __init__(
        self,
        model_name:      str,
        num_labels:      int   = 2,
        dropout:         float = 0.1,
        use_soft_labels: bool  = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.use_soft_labels = use_soft_labels
        hidden = self.encoder.config.hidden_size

        # 3 positions: root, parent, tweet
        self.position_embeddings = nn.Embedding(3, hidden)

        self.thread_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward sublayer after attention (standard Transformer block)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token representations for a batch of texts."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(output.last_hidden_state, attention_mask)  # (B, H)

    def forward(
        self,
        root_input_ids:        torch.Tensor,
        root_attention_mask:   torch.Tensor,
        parent_input_ids:      torch.Tensor,
        parent_attention_mask: torch.Tensor,
        tweet_input_ids:       torch.Tensor,
        tweet_attention_mask:  torch.Tensor,
        labels:                Optional[torch.Tensor] = None,
        soft_labels:           Optional[torch.Tensor] = None,
    ) -> dict:
        root_repr   = self.encode(root_input_ids,   root_attention_mask)    # (B, H)
        parent_repr = self.encode(parent_input_ids, parent_attention_mask)  # (B, H)
        tweet_repr  = self.encode(tweet_input_ids,  tweet_attention_mask)   # (B, H)

        # Stack into thread sequence: (B, 3, H)
        positions = torch.arange(3, device=tweet_repr.device)
        pos_emb   = self.position_embeddings(positions).unsqueeze(0)  # (1, 3, H)
        thread    = torch.stack([root_repr, parent_repr, tweet_repr], dim=1) + pos_emb

        # Mask padding-only context positions
        root_empty   = (root_attention_mask.sum(dim=1)   <= 2)
        parent_empty = (parent_attention_mask.sum(dim=1) <= 2)
        tweet_empty  = torch.zeros(tweet_repr.shape[0], dtype=torch.bool, device=tweet_repr.device)
        key_padding_mask = torch.stack([root_empty, parent_empty, tweet_empty], dim=1)  # (B, 3)

        # Tweet queries the full thread (root + parent + itself)
        # Information flow: tweet → attends to → context
        attn_out, _ = self.thread_attention(
            query=thread[:, 2:, :],  # tweet as query  (B, 1, H)
            key=thread,              # full thread      (B, 3, H)
            value=thread,
            key_padding_mask=key_padding_mask,
        )  # (B, 1, H)

        # Residual + norm
        tweet_hidden = self.norm1(tweet_repr + attn_out.squeeze(1))  # (B, H)

        # Feed-forward sublayer
        tweet_hidden = self.norm2(tweet_hidden + self.ffn(tweet_hidden))  # (B, H)

        logits = self.classifier(tweet_hidden)  # (B, num_labels)

        loss = None
        if labels is not None:
            if self.use_soft_labels and soft_labels is not None:
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = nn.KLDivLoss(reduction="batchmean")(log_probs, soft_labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


class CrossAttentionContextModel(nn.Module):
    """
    Token-level cross-attention model.

    Every tweet token attends to every context token (root + parent
    concatenated), capturing fine-grained interactions between the tweet
    and its conversational context. level4 (hoax) is intentionally excluded.
    """

    def __init__(
        self,
        model_name:       str,
        num_labels:       int   = 2,
        hidden_dim:       int   = 384,
        num_heads:        int   = 8,
        num_cross_layers: int   = 2,
        dropout:          float = 0.1,
        use_soft_labels:  bool  = False,
    ) -> None:
        super().__init__()

        self.encoder         = AutoModel.from_pretrained(model_name)
        self.num_labels      = num_labels
        self.use_soft_labels = use_soft_labels
        hidden = self.encoder.config.hidden_size  # 768

        # Stack of cross-attention + FFN layers
        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=hidden,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm1": nn.LayerNorm(hidden),
                "ffn": nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 4, hidden),
                ),
                "norm2": nn.LayerNorm(hidden),
            })
            for _ in range(num_cross_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return all token embeddings for a batch of texts."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state  # (B, seq_len, H)

    def forward(
        self,
        context_input_ids:        torch.Tensor,
        context_attention_mask:   torch.Tensor,
        tweet_input_ids:          torch.Tensor,
        tweet_attention_mask:     torch.Tensor,
        labels:                   Optional[torch.Tensor] = None,
        soft_labels:              Optional[torch.Tensor] = None,
        class_weights:            Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> dict:
        """
        Args:
            context_input_ids:        (B, ctx_len)  root + parent concatenated
            context_attention_mask:   (B, ctx_len)
            tweet_input_ids:          (B, tweet_len)
            tweet_attention_mask:     (B, tweet_len)
            labels:                   (B,) hard labels, optional
            soft_labels:              (B, num_labels) soft labels, optional
            class_weights:            (num_labels,) for class imbalance, optional
            return_attention_weights: return last-layer attention weights
        """
        context_tokens = self.encode(context_input_ids, context_attention_mask)  # (B, ctx_len, H)
        tweet_tokens   = self.encode(tweet_input_ids,   tweet_attention_mask)    # (B, tweet_len, H)

        context_key_padding_mask = (context_attention_mask == 0)  # (B, ctx_len)

        last_attn_weights = None
        enriched = tweet_tokens

        for layer in self.cross_layers:
            # Tweet tokens attend to context tokens
            attended, attn_weights = layer["cross_attn"](
                query=enriched,
                key=context_tokens,
                value=context_tokens,
                key_padding_mask=context_key_padding_mask,
                need_weights=return_attention_weights,
                average_attn_weights=False,
            )
            enriched = layer["norm1"](enriched + attended)              # residual + norm
            enriched = layer["norm2"](enriched + layer["ffn"](enriched))  # FFN + norm
            last_attn_weights = attn_weights

        # Mean pool over non-padding tweet tokens
        pooled = mean_pool(enriched, tweet_attention_mask)  # (B, H)
        logits = self.classifier(pooled)                    # (B, num_labels)

        loss = None
        if labels is not None:
            if self.use_soft_labels and soft_labels is not None:
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = nn.KLDivLoss(reduction="batchmean")(log_probs, soft_labels)
            else:
                loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)

        out: dict = {"logits": logits, "loss": loss}
        if return_attention_weights:
            out["attention_weights"] = last_attn_weights  # (B, heads, tweet_len, ctx_len)
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