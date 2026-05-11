"""
Load data from the .csv files
"""

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

    return df_train, df_test