from sklearn.ensemble import GradientBoostingRegressor

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


def GBM(df, sub1_column, sub2_column, condition_id_column, product_column):
    scatter_dir, _, metrics_path, model_path = setup_output_paths("GBM")
    plotter = create_plotter(scatter_dir, color_theme="style5")

    splits = prepare_dataset_splits(df)
    features, _ = build_features_for_splits(
        splits, sub1_column, sub2_column, condition_id_column, product_column
    )
    targets = extract_targets(splits)

    prefix = "Buchwald GBM"
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )

    print("Training GBM model...")
    model.fit(features["train"], targets["train"])

    metrics_dict: dict = {}
    evaluate_model(
        model,
        features["train"],
        targets["train"],
        "Training",
        metrics_dict,
        plotter,
        plot_title_prefix=prefix,
    )
    print("Validating on validation set...")
    validate_model(
        model,
        features["val"],
        targets["val"],
        "Validation",
        metrics_dict,
        plotter,
        plot_title_prefix=prefix,
    )

    print("\nEvaluating model...")
    evaluate_model(model, features["random"], targets["random"], "Random Split", metrics_dict, plotter, plot_title_prefix=prefix)
    evaluate_model(model, features["partial novelty"], targets["partial novelty"], "Partial Novelty", metrics_dict, plotter, plot_title_prefix=prefix)
    evaluate_model(model, features["full novelty"], targets["full novelty"], "Full Novelty", metrics_dict, plotter, plot_title_prefix=prefix)

    dump_metrics(metrics_dict, metrics_path)
    persist_object(model, model_path)
    summarize_metrics(metrics_dict)

    return model, metrics_dict
