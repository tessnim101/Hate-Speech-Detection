"""
Training configuration.
"""

CONFIG = {
    "model_name":"cardiffnlp/twitter-xlm-roberta-base",
    "max_len": 192,
    "max_len_fusion": 512,
    "max_len_context": 256,  
    "max_len_tweet": 128,    
    "train_bs": 16,
    "eval_bs": 16,
    "lr": 2e-5,             # overwritten per model in training/runners.py
    "weight_decay": 0.0025, # overwritten per model in training/runners.py
    "epochs": 8,
    "output_dir": "./results",
    "log_dir": "./logs",
}