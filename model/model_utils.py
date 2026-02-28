import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from one_hot import create_one_hot_encoding
from plot_utils import PlotUtils


SPLIT_DISPLAY_NAMES: List[Tuple[str, str]] = [
    ("train", "Training"),
    ("val", "Validation"),
    ("random", "Random Split"),
    ("partial novelty", "Partial Novelty"),
    ("full novelty", "Full Novelty"),
]


def _log_split_sizes(splits: Dict[str, pd.DataFrame]) -> None:
    for key, label in SPLIT_DISPLAY_NAMES:
        print(f"{label} set size: {len(splits[key])}")


def prepare_dataset_splits(df: pd.DataFrame, class_column: str = "class") -> Dict[str, pd.DataFrame]:
    splits = {
        key: df[df[class_column] == key].copy()
        for key, _ in SPLIT_DISPLAY_NAMES
    }
    _log_split_sizes(splits)
    return splits


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        return np.array(fp)

    return np.zeros(n_bits)


def get_fingerprint_features(
    df: pd.DataFrame,
    sub_1_col: str,
    sub_2_col: str,
    condition_id_column: str,
    prod_col: str,
    condition_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    sub_1_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df[sub_1_col]])
    sub_2_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df[sub_2_col]])
    prod_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df[prod_col]])

    condition_features = create_one_hot_encoding(df, condition_id_column, total_cols=96)

    if condition_columns is not None:
        for col in condition_columns:
            if col not in condition_features:
                condition_features[col] = 0
        condition_features = condition_features[list(condition_columns)]
        used_columns = list(condition_columns)
    else:
        used_columns = condition_features.columns.tolist()

    features = np.hstack([
        sub_1_fp,
        sub_2_fp,
        condition_features.to_numpy(dtype=float),
        prod_fp,
    ])

    return features, used_columns


def build_features_for_splits(
    splits: Dict[str, pd.DataFrame],
    sub1_col: str,
    sub2_col: str,
    condition_id_column: str,
    product_column: str,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    features: Dict[str, np.ndarray] = {}
    condition_columns: Optional[List[str]] = None

    for key, _ in SPLIT_DISPLAY_NAMES:
        df = splits[key]
        if key == "train":
            X, condition_columns = get_fingerprint_features(
                df, sub1_col, sub2_col, condition_id_column, product_column, condition_columns=None
            )
        else:
            X, _ = get_fingerprint_features(
                df, sub1_col, sub2_col, condition_id_column, product_column, condition_columns=condition_columns
            )
        features[key] = X

    return features, condition_columns or []


def extract_targets(splits: Dict[str, pd.DataFrame], target_column: str = "yield") -> Dict[str, np.ndarray]:
    return {
        key: splits[key][target_column].values
        for key, _ in SPLIT_DISPLAY_NAMES
    }


def setup_output_paths(
    model_name: str,
    scatter_root: str = "buchwald_scatter_output", 
    model_root: str = "buchwald_output",
) -> Tuple[str, str, str, str]:
    scatter_dir = os.path.join(scatter_root, model_name)
    os.makedirs(scatter_dir, exist_ok=True)

    model_dir = model_root
    os.makedirs(model_dir, exist_ok=True)

    metrics_path = os.path.join(scatter_dir, f"{model_name}_metrics.json")
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")

    return scatter_dir, model_dir, metrics_path, model_path


def create_plotter(scatter_dir: str, color_theme: str) -> PlotUtils:
    return PlotUtils(output_dir=scatter_dir, color_theme=color_theme)


def dump_metrics(metrics_dict: Dict[str, dict], metrics_path: str) -> str:
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    return metrics_path


def persist_object(obj, artifact_path: str) -> str:
    joblib.dump(obj, artifact_path)
    print(f"Saved artifact to: {artifact_path}")
    return artifact_path


def summarize_metrics(metrics_dict: Dict[str, dict], title: str = "MODEL EVALUATION SUMMARY") -> pd.DataFrame:
    if not metrics_dict:
        print("No metrics to summarize.")
        return pd.DataFrame()

    summary_data = []
    for set_name, metrics in metrics_dict.items():
        summary_data.append({
            "Dataset": set_name,
            "Samples": metrics["num_samples"],
            "MSE": f"{metrics['MSE']:.4f}",
            "RMSE": f"{metrics['RMSE']:.4f}",
            "MAE": f"{metrics['MAE']:.4f}",
            "R2": f"{metrics['R2']:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(summary_df.to_string(index=False))
    return summary_df


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "num_samples": len(y_true),
    }


def _print_metrics(set_name: str, metrics: dict) -> None:
    print(f"---- {set_name} ----")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Mean Squared Error: {metrics['MSE']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R-squared (R2): {metrics['R2']:.4f}")


def _plot_scores(
    plotter: PlotUtils,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    set_name: str,
    metrics: dict,
    title_prefix: str = "Buchwald Regression",
) -> None:
    plotter.plot_scatter(
        y_true,
        y_pred,
        metrics["MSE"],
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
        xlabel="Actual Yield",
        ylabel="Predicted Yield",
        title_prefix=title_prefix,
        dataset_name=set_name,
    )


def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    set_name: str,
    metrics_dict: Dict[str, dict],
    plotter: PlotUtils,
    plot_title_prefix: str = "Buchwald Regression",
) -> Optional[np.ndarray]:
    if len(y_true) == 0:
        print(f"---- {set_name} ----")
        print("No samples to evaluate.")
        return None

    y_pred = model.predict(X)
    metrics = _compute_metrics(y_true, y_pred)
    metrics_dict[set_name] = metrics

    _print_metrics(set_name, metrics)
    _plot_scores(plotter, y_true, y_pred, set_name, metrics, plot_title_prefix)

    return y_pred


def validate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    set_name: str,
    metrics_dict: Dict[str, dict],
    plotter: PlotUtils,
    plot_title_prefix: str = "Buchwald Regression",
) -> Optional[np.ndarray]:
    if len(y_val) == 0:
        print(f"---- {set_name} ----")
        print("No samples to evaluate.")
        return None

    y_pred = model.predict(X_val)
    metrics = _compute_metrics(y_val, y_pred)
    metrics_dict[set_name] = metrics

    print("Validation Results:")
    print(f"Mean Squared Error: {metrics['MSE']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R-squared (R2): {metrics['R2']:.4f}")

    _plot_scores(plotter, y_val, y_pred, set_name, metrics, plot_title_prefix)

    return y_pred
