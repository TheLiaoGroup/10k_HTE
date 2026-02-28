# 10k_HTE Modeling

## Environment Setup

```bash
conda create -n 10k_HTE python=3.9 -y
conda activate 10k_HTE
pip install -r requirements.txt
```

Once the dependencies are installed the environment is ready for data preparation and training.

## Data

All required datasets are stored in the `data` folder and include the following collections:

- `Acylation`
- `Buchwald`
- `Suzuki`

## Data Splitting

Use the helper in `data_split.py` to partition a dataset into training, validation, and test splits:

```python
from data_split import data_split

data_split(
    csv_path,
    sub1_column,
    sub2_column,
    test_size=0.1,
    val_size=0.1,
    novelty_size=0.1,
    random_state=42,
    save_to_csv=True,
    output_path=None,
)
```

The split assigns samples to the following classes:

- `train`: training subset
- `val`: validation subset
- `random`: random split test set
- `partial novelty`: partial substrate novelty test set
- `full novelty`: full substrate novelty test set

## Condition Encoding

Each reaction condition is encoded as a unique integer identifier in the range `1–96`. The one-hot helper provides this transformation:

```python
condition_features = create_one_hot_encoding(df, condition_id_column, total_cols=96)
```

## Model Training and Evaluation

Use `train_machine_learning_model` to train any of the supported model types. Pass the model name together with the DataFrame and relevant column names:

```python
from model.RandomForest import RF
from models import train_machine_learning_model

train_machine_learning_model(
    'RandomForest',
    df,
    'sub_1_smiles',
    'sub_2_smiles',
    'condition_id',
    'product_smiles',
)
```

Supported model types include `Ridge`, `Lasso`, `GBM`, `XGBoost`, `SVR`, and `RandomForest`.

## Citation

If you use this data/code/model, please cite the relevant papers associated with the project.

## License

This project is licensed under the [MIT License](LICENSE).
