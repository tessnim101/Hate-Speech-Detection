"""
Configuration
"""

CONFIG = {
    "model_name": "xlm-roberta-base",
    "max_len": 128,
    "max_len_context": 512,
    "train_bs": 16,
    "eval_bs": 16,
    "lr": 2e-5,
    #"warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "epochs": 5,
    "output_dir": "./results",
    "log_dir": "./logs",
}