"""
Training configuration.
"""

CONFIG = {
    "model_name":    "xlm-roberta-base",
    "max_len":       192,
    "max_len_context": 256,   # context for cross-attention model
    "max_len_tweet": 128,     # tweet for cross-attention model
    "train_bs":      8,
    "eval_bs":       8,
    "learning_rate": 1e-5,    
    "weight_decay":  0.01,    
    "warmup_ratio":  0.1,   
    "max_grad_norm": 1.0,     
    "epochs":        9,
    "early_stopping_patience": 3,
    "output_dir":    "./results",
    "log_dir":       "./logs",
    "imbalance_strategy": "weighted_loss",  # "weighted_loss", "oversample", "focal_loss"
    "focal_gamma":        2.0,
}