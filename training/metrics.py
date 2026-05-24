"""
Compute metrics during training/validation
"""

from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_binary": f1_score(labels, preds),
        "f1_class0": f1_score(labels, preds, pos_label=0),
        "f1_class1": f1_score(labels, preds, pos_label=1),
    }
