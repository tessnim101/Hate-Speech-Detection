"""
Pre-process data, that is replace parent and root tweet id by actual textual tweet
Also reduce size of training and testing data if training on CPU
"""

import pandas as pd

def stratified_sample(df, n):
    groups = []
    for _, group in df.groupby('stereotype'):
        groups.append(group.sample(min(len(group), n // 2), random_state=42))
    return pd.concat(groups).reset_index(drop=True)


def ids_to_text(df):
    id_to_text = df[['comment_id', 'text']].rename(
        columns={'comment_id': 'lookup_id', 'text': 'lookup_text'}
    )

    df = df.merge(
        id_to_text,
        left_on='level2',
        right_on='lookup_id',
        how='left'
    ).rename(columns={'lookup_text': 'parent_text'}).drop(columns=['lookup_id'])

    df = df.merge(
        id_to_text,
        left_on='level3',
        right_on='lookup_id',
        how='left'
    ).rename(columns={'lookup_text': 'root_text'}).drop(columns=['lookup_id'])

    df['parent_text'] = df['parent_text'].fillna("")
    df['root_text'] = df['root_text'].fillna("")

    return df