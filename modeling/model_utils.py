"""
Load model
"""

from transformers import AutoModelForSequenceClassification


def load_model(model_name, num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
