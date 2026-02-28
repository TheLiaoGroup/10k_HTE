import pandas as pd
import numpy as np
import re


def create_one_hot_encoding(df, condition_id_col, total_cols=96):

    
    df[condition_id_col] = pd.to_numeric(df[condition_id_col], errors='coerce').fillna(0).astype(int)
    
    
    invalid_values = df[~df[condition_id_col].between(1, 96) & (df[condition_id_col] != 0)][condition_id_col].unique()
    if len(invalid_values) > 0:
        print(f"警告：存在超出1-96范围的值: {invalid_values}")
        df.loc[~df[condition_id_col].between(1, 96) & (df[condition_id_col] != 0), condition_id_col] = 0
    

    n_samples = len(df)
    one_hot_matrix = np.zeros((n_samples, total_cols), dtype=int)
    
    condition_indices = df[condition_id_col] - 1
    
    valid_mask = (condition_indices >= 0) & (condition_indices < 96)
    valid_indices = condition_indices[valid_mask].astype(int)
    valid_rows = np.arange(n_samples)[valid_mask]
    
    one_hot_matrix[valid_rows, valid_indices] = 1
    
    one_hot_df = pd.DataFrame(
        one_hot_matrix,
        columns=[f'condition_{i+1}' for i in range(total_cols)],
        index=df.index
    )
    
    return one_hot_df


