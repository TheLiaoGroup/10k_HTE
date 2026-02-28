from sklearn.ensemble import RandomForestRegressor

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


def RF(df, sub1_column, sub2_column, condition_id_column, product_column):
    scatter_dir, _, metrics_path, model_path = setup_output_paths("RandomForestRegressor")
    plotter = create_plotter(scatter_dir, color_theme="style4")

    splits = prepare_dataset_splits(df)
    features, _ = build_features_for_splits(
        splits, sub1_column, sub2_column, condition_id_column, product_column
    )
    targets = extract_targets(splits)

    X_train = features["train"]
    X_val = features["val"]
    X_random = features["random"]
    X_partial = features["partial novelty"]
    X_full = features["full novelty"]

    y_train = targets["train"]
    y_val = targets["val"]
    y_random = targets["random"]
    y_partial = targets["partial novelty"]
    y_full = targets["full novelty"]

    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    metrics_dict: dict = {}
    prefix = "Buchwald RandomForest Regression"
    evaluate_model(model, X_train, y_train, "Training", metrics_dict, plotter, plot_title_prefix=prefix)
    print("Validating on validation set...")
    validate_model(model, X_val, y_val, "Validation", metrics_dict, plotter, plot_title_prefix=prefix)

    print("\nEvaluating model...")
    evaluate_model(model, X_random, y_random, "Random Split", metrics_dict, plotter, plot_title_prefix=prefix)
    evaluate_model(model, X_partial, y_partial, "Partial Novelty", metrics_dict, plotter, plot_title_prefix=prefix)
    evaluate_model(model, X_full, y_full, "Full Novelty", metrics_dict, plotter, plot_title_prefix=prefix)

    dump_metrics(metrics_dict, metrics_path)
    persist_object(model, model_path)
    summarize_metrics(metrics_dict)

    return model, metrics_dict
