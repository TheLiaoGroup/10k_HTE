import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from model.model_utils import (
    build_features_for_splits,
    create_plotter,
    dump_metrics,
    evaluate_model,
    extract_targets,
    prepare_dataset_splits,
    persist_object,
    setup_output_paths,
    summarize_metrics,
    validate_model,
)


def SVM(df, sub1_column, sub2_column, condition_id_column, product_column):
    scatter_dir, model_dir, metrics_path, model_path = setup_output_paths("SVM")
    plotter = create_plotter(scatter_dir, color_theme="style3")

    splits = prepare_dataset_splits(df)
    features, _ = build_features_for_splits(
        splits, sub1_column, sub2_column, condition_id_column, product_column
    )
    targets = extract_targets(splits)

    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(features["train"])
    X_val_s = scaler.transform(features["val"])
    X_random_s = scaler.transform(features["random"])
    X_partial_s = scaler.transform(features["partial novelty"])
    X_full_s = scaler.transform(features["full novelty"])

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=50.0,
                epsilon=0.001,
                gamma=0.005,
            ),
        ),
    ])

    print("Training SVM...")
    model.fit(X_train_s, targets["train"])

    metrics_dict: dict = {}
    prefix = "Buchwald SVR"
    evaluate_model(model, X_train_s, targets["train"], "Training", metrics_dict, plotter, plot_title_prefix=prefix)
    print("Validating on validation set...")
    validate_model(model, X_val_s, targets["val"], "Validation", metrics_dict, plotter, plot_title_prefix=prefix)

    print("\nEvaluating model...")
    evaluate_model(model, X_random_s, targets["random"], "Random Split", metrics_dict, plotter, plot_title_prefix=prefix)
    evaluate_model(model, X_partial_s, targets["partial novelty"], "Partial Novelty", metrics_dict, plotter, plot_title_prefix=prefix)
    evaluate_model(model, X_full_s, targets["full novelty"], "Full Novelty", metrics_dict, plotter, plot_title_prefix=prefix)

    dump_metrics(metrics_dict, metrics_path)
    persist_object(model, model_path)
    scaler_path = os.path.join(model_dir, "svm_scaler.pkl")
    persist_object(scaler, scaler_path)
    summarize_metrics(metrics_dict)

    return model, scaler, metrics_dict
