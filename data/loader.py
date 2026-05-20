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
        bt = pd.read_csv(bt_path)[["comment_id", "backtranslated_text"]]
        df_train = df_train.merge(bt, on="comment_id", how="left")
        df_train["backtranslated_text"] = df_train["backtranslated_text"].fillna("")
    else:
        df_train["backtranslated_text"] = ""

    df_test["backtranslated_text"] = ""

    return df_train, df_test