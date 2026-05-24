"""
Load data from the .csv files
"""

import os
import pandas as pd


def load_data(path_data):
    df_train = pd.read_csv(path_data + 'train.csv')
    df_test  = pd.read_csv(path_data + 'test.csv')

    df_train = df_train[df_train['source'] == 'stereohoax']
    df_test  = df_test[df_test['source']  == 'stereohoax']

    hoaxes = pd.read_csv(path_data + 'hoaxes.csv', header=1).rename(
        columns={'id': 'level4', 'text': 'hoax'}
    )
    df_train = df_train.merge(hoaxes, on='level4', how='inner')
    df_test  = df_test.merge(hoaxes,  on='level4', how='inner')

    # Back-translations are a training-only augmentation
    bt_path = path_data + "train_bt.csv"
    if os.path.exists(bt_path):
        bt = pd.read_csv(bt_path)
        bt_cols = [c for c in bt.columns if c == "comment_id" or c.startswith("bt_")]
        df_train = df_train.merge(bt[bt_cols], on="comment_id", how="left")
        for col in bt_cols:
            if col != "comment_id":
                df_train[col] = df_train[col].fillna("")
    else:
        for col in ["bt_text", "bt_hoax", "bt_parent_text", "bt_root_text"]:
            df_train[col] = ""

    df_test["backtranslated_text"] = ""

    return df_train, df_test