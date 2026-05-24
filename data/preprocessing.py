"""
Pre-process data
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS_ES = set(stopwords.words("spanish"))


# Text cleaning
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean a single string:
      - Lowercase
      - Remove URLs
      - Remove mentions (@user) and hashtag symbols (keep word)
      - Remove punctuation
      - Collapse whitespace
      - Optionally remove Spanish stopwords

    Note: stopword removal is OFF by default for context fields (root/parent)
    because function words can carry pragmatic cues relevant to stereotype
    detection. Turn it on only for ablation experiments.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                      # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # hashtags → word
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\s+", " ", text).strip()              # whitespace

    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
        text = " ".join(tokens)

    return text


def clean_df(df: pd.DataFrame, remove_stopwords: bool = False) -> pd.DataFrame:
    """Apply clean_text to all text columns in-place."""
    for col in ["text", "parent_text", "root_text"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_text(x, remove_stopwords))
    return df


# Filtering & splitting
def filter_contextual_tweets(df):
    """
    Keep only tweets that have both a parent and a root tweet.

    Tweets without context (level2 == 0 or level3 == 0) are dropped so that
    the baseline and hierarchical models are evaluated on the same population.
    """
    before = len(df)
    df = df[(df["level2"] != "0") & (df["level3"] != "0")].reset_index(drop=True)
    after = len(df)
    print(f"[filter] Kept {after:,} / {before:,} tweets with full context "
          f"({before - after:,} dropped).")
    return df


def stratified_sample(df, n):
    """Reduce dataset size for CPU-friendly tests."""
    groups = []
    for _, group in df.groupby("stereotype"):
        groups.append(group.sample(min(len(group), n // 2), random_state=42))
    return pd.concat(groups).reset_index(drop=True)


def split_train_validation(df, val_size=0.2):
    """Split training data into training and validation."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["stereotype"],
        random_state=42,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# Context resolution
def ids_to_text(df):
    """Replace parent and root tweet ids with their actual text."""
    id_to_text = df[["comment_id", "text"]].rename(
        columns={"comment_id": "lookup_id", "text": "lookup_text"}
    )

    df = df.merge(
        id_to_text, left_on="level2", right_on="lookup_id", how="left"
    ).rename(columns={"lookup_text": "parent_text"}).drop(columns=["lookup_id"])

    df = df.merge(
        id_to_text, left_on="level3", right_on="lookup_id", how="left"
    ).rename(columns={"lookup_text": "root_text"}).drop(columns=["lookup_id"])

    df["parent_text"] = df["parent_text"].fillna("")
    df["root_text"]   = df["root_text"].fillna("")

    return df


# Tokenizers
def tokenize_baseline(dataset, tokenizer, max_len):
    """Tokenize for the baseline model (tweet only, no context)."""
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
    return dataset.map(tokenize, batched=True)


def tokenize_hierarchical(dataset, tokenizer, max_len: int):
    """
    Tokenize for the hierarchical model.
    Encodes root, parent, and tweet independently.
    """
    def tokenize(example):
        def enc(texts):
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )

        root   = enc(example["root_text"])
        parent = enc(example["parent_text"])
        tweet  = enc(example["text"])

        return {
            "root_input_ids":        root["input_ids"],
            "root_attention_mask":   root["attention_mask"],
            "parent_input_ids":      parent["input_ids"],
            "parent_attention_mask": parent["attention_mask"],
            "tweet_input_ids":       tweet["input_ids"],
            "tweet_attention_mask":  tweet["attention_mask"],
        }

    return dataset.map(tokenize, batched=True)


def tokenize_cross_attention(
    dataset,
    tokenizer,
    max_len_tweet:   int,
    max_len_context: int,
):
    """
    Tokenize for the cross-attention model.

    Tweet and context (root + parent) are tokenized separately with
    different max lengths — context gets more room since it concatenates
    two texts, tweet stays short since it is a single sentence.

    Args:
        dataset:         HuggingFace Dataset with "text", "parent_text", "root_text".
        tokenizer:       Matching tokenizer.
        max_len_tweet:   Max tokens for the tweet (typically 128).
        max_len_context: Max tokens for the concatenated context (typically 256).
    """
    def tokenize(example):
        context = [
            r + tokenizer.sep_token + p
            for r, p in zip(example["root_text"], example["parent_text"])
        ]
        ctx_enc   = tokenizer(context,         padding="max_length", truncation=True, max_length=max_len_context)
        tweet_enc = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_len_tweet)

        return {
            "context_input_ids":      ctx_enc["input_ids"],
            "context_attention_mask": ctx_enc["attention_mask"],
            "tweet_input_ids":        tweet_enc["input_ids"],
            "tweet_attention_mask":   tweet_enc["attention_mask"],
        }

    return dataset.map(tokenize, batched=True)


# Augmentation
def augment_with_backtranslation(df):
    """
    Expand the training set by appending back-translated tweets as new rows.

    For each row where backtranslated_text is non-empty, a copy of the row is
    created with text replaced by the back-translated version and the same label.
    """
    bt_rows = df[df["backtranslated_text"].ne("")].copy()
    bt_rows["text"] = bt_rows["backtranslated_text"]
    augmented = pd.concat([df, bt_rows], ignore_index=True)
    print(f"[augment] Added {len(bt_rows):,} back-translated rows → {len(augmented):,} total.")
    return augmented