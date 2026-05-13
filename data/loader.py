"""
Load data from the .csv files
"""

import os
import pandas as pd


def load_data(path_data):
    df_train = pd.read_csv(path_data + 'train.csv')
    df_test = pd.read_csv(path_data + 'test.csv')

    # Keep only stereohoax
    df_train = df_train[df_train['source'] == 'stereohoax']
    df_test = df_test[df_test['source'] == 'stereohoax']

    # Load hoaxes
    hoaxes = pd.read_csv(path_data + 'hoaxes.csv', header=1).rename(
        columns={'id': 'level4', 'text': 'hoax'}
    )

    df_train = df_train.merge(hoaxes, on='level4', how='inner')
    df_test = df_test.merge(hoaxes, on='level4', how='inner')

    # Merge pre-computed back-translations when available (produced by translate.py)
    for split, df in [("train", df_train), ("test", df_test)]:
        bt_path = path_data + f"{split}_bt.csv"
        if os.path.exists(bt_path):
            bt = pd.read_csv(bt_path)[["comment_id", "backtranslated_text"]]
            df_merged = df.merge(bt, on="comment_id", how="left")
            df_merged["backtranslated_text"] = df_merged["backtranslated_text"].fillna("")
            if split == "train":
                df_train = df_merged
            else:
                df_test = df_merged
        else:
            df["backtranslated_text"] = ""

    return df_train, df_test