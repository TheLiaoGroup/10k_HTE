import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def data_split(csv_path,sub1_column,sub2_column,test_size=0.1,val_size=0.1,novelty_size=0.1,
    random_state=42,
    save_to_csv=True,
    output_path=None
):

    df = pd.read_csv(csv_path)
    print(f"Original data size: {df.shape}")

    df['class'] = ''

    indices = np.arange(len(df))


    # 1. Random split
    train_val_idx, random_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    val_relative_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_size,
        random_state=random_state,
        shuffle=True
    )

    df.loc[train_idx, 'class'] = 'train'
    df.loc[val_idx, 'class'] = 'val'
    df.loc[random_idx, 'class'] = 'random'

  
    # 2. Novelty split
    random_df = df[df['class'] == 'random']


    unique_smiles = set(random_df[sub1_column]).union(
        set(random_df[sub2_column])
    )

    train_smiles, novelty_smiles = train_test_split(
        list(unique_smiles),
        test_size=novelty_size,
        random_state=random_state
    )

    train_smiles = set(train_smiles)
    novelty_smiles = set(novelty_smiles)

    # full novelty
    full_novelty_mask = (
        random_df[sub1_column].isin(novelty_smiles) &
        random_df[sub2_column].isin(novelty_smiles)
    )

    # partial novelty
    partial_novelty_mask = (
        random_df[sub1_column].isin(novelty_smiles) ^
        random_df[sub2_column].isin(novelty_smiles)
    )

  
    df.loc[random_df[full_novelty_mask].index, 'class'] = 'full novelty'
    df.loc[random_df[partial_novelty_mask].index, 'class'] = 'partial novelty'



    for c in ['train', 'val', 'random', 'partial novelty', 'full novelty']:
        n = (df['class'] == c).sum()
        print(f"{c:16s}: {n:5d} 条 ({n / len(df) * 100:.1f}%)")

  

    if save_to_csv:
        if output_path is None:
            base_name = csv_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_split.csv"

        df.to_csv(output_path, index=False)

        for c in ['train', 'val', 'random', 'partial novelty', 'full novelty']:
            sub_df = df[df['class'] == c]
            sub_df.to_csv(
                f"{output_path.rsplit('.', 1)[0]}_{c.replace(' ', '_')}.csv",
                index=False
            )

        print(f"\n data has save to: {output_path}")

    return df
