import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
import joblib


FEATURES_PATH = "data/processed/match_features.csv"
MODEL_PATH = "outputs/xgb_model.pkl"
FEATURE_COLUMNS_PATH = "outputs/xgb_feature_columns.pkl"
PURE_MODEL_PATH = "outputs/xgb_model_pure_tennis.pkl"
PURE_FEATURE_COLUMNS_PATH = "outputs/xgb_feature_columns_pure_tennis.pkl"
MARKET_MODEL_PATH = "outputs/xgb_model_market_aware.pkl"
MARKET_FEATURE_COLUMNS_PATH = "outputs/xgb_feature_columns_market_aware.pkl"
METRICS_PATH = "outputs/xgb_training_metrics.csv"
PARAM_SEARCH_RESULTS_PATH = "outputs/xgb_param_search_results.csv"
FORCE_TUNING_ENV = "FORCE_XGB_TUNING"
XGB_SEARCH_ITER_ENV = "XGB_SEARCH_ITER"
SKIP_TUNING_ENV = "SKIP_XGB_TUNING"
DEFAULT_PARAMS = {
    "subsample": 0.85,
    "reg_lambda": 2,
    "reg_alpha": 0.01,
    "n_estimators": 800,
    "min_child_weight": 8,
    "max_depth": 4,
    "learning_rate": 0.01,
    "gamma": 0.1,
    "colsample_bytree": 0.65,
}
SURFACE_COLUMNS = ("surface_Hard", "surface_Clay", "surface_Grass", "surface_Carpet")


PARAM_GRID = {
    "n_estimators": [500, 800, 1100, 1400],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.06],
    "subsample": [0.65, 0.75, 0.85, 0.95],
    "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
    "min_child_weight": [1, 3, 5, 8, 12],
    "gamma": [0, 0.05, 0.1, 0.2, 0.4],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.3],
    "reg_lambda": [0.5, 1, 1.5, 2, 3, 5],
}


def load_saved_training_state(model_name, feature_count):
    if os.getenv(FORCE_TUNING_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return None

    metrics_path = Path(METRICS_PATH)
    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path)
    matches = metrics[
        (metrics["model_name"] == model_name)
        & (metrics["feature_count"] == feature_count)
    ]
    if matches.empty:
        return None

    params_text = matches.iloc[-1].get("best_params")
    if pd.isna(params_text):
        return None

    try:
        best_params = ast.literal_eval(params_text)
    except (ValueError, SyntaxError):
        return None

    cv_log_loss = matches.iloc[-1].get("best_cv_log_loss")
    if pd.isna(cv_log_loss):
        search_results_path = Path(PARAM_SEARCH_RESULTS_PATH)
        if search_results_path.exists():
            search_results = pd.read_csv(search_results_path)
            search_matches = search_results[
                search_results["model_name"] == model_name
            ]
            if not search_matches.empty:
                best_search_row = search_matches.sort_values("rank_test_score").iloc[0]
                cv_log_loss = -best_search_row["mean_test_score"]

    return {
        "best_params": best_params,
        "best_cv_log_loss": cv_log_loss,
    }


def load_last_params(model_name):
    metrics_path = Path(METRICS_PATH)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        matches = metrics[metrics["model_name"] == model_name]
        if not matches.empty:
            params_text = matches.iloc[-1].get("best_params")
            if not pd.isna(params_text):
                try:
                    return ast.literal_eval(params_text)
                except (ValueError, SyntaxError):
                    pass

    return DEFAULT_PARAMS.copy()


def make_xgb_model(params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )


def make_calibrated_model(base_model, cv=3):
    try:
        return CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_model,
            method="sigmoid",
            cv=cv,
        )


def make_prefit_calibrator(base_model):
    try:
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method="sigmoid",
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=FrozenEstimator(base_model),
            method="sigmoid",
        )


def calibration_sample(X, y, max_rows=15000):
    if len(X) <= max_rows:
        return X, y

    sample = (
        pd.DataFrame({"target": y}, index=X.index)
        .groupby("target", group_keys=False)
        .apply(
            lambda group: group.sample(
                min(len(group), max_rows // 2),
                random_state=42,
            )
        )
        .index
    )
    return X.loc[sample], y.loc[sample]


class SurfaceCalibratedModel:
    def __init__(self, default_model, surface_models, surface_columns):
        self.default_model = default_model
        self.surface_models = surface_models
        self.surface_columns = surface_columns

    def _model_for_row(self, row):
        for surface_col in self.surface_columns:
            if surface_col in row.index and row[surface_col] == 1:
                return self.surface_models.get(surface_col, self.default_model)
        return self.default_model

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        probabilities = []
        for _, row in X.iterrows():
            model = self._model_for_row(row)
            row_df = pd.DataFrame([row], columns=X.columns)
            probabilities.append(model.predict_proba(row_df)[0])

        return np.array(probabilities)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_surface_calibrated_model(params, X, y, min_surface_rows=300):
    base_model = make_xgb_model(params)
    base_model.fit(X, y)

    cal_X, cal_y = calibration_sample(X, y)
    default_model = make_prefit_calibrator(base_model)
    default_model.fit(cal_X, cal_y)

    surface_models = {}
    for surface_col in SURFACE_COLUMNS:
        if surface_col not in X.columns:
            continue

        surface_mask = X[surface_col] == 1
        if surface_mask.sum() < min_surface_rows:
            continue

        calibrator = make_prefit_calibrator(base_model)
        surface_X, surface_y = calibration_sample(
            X.loc[surface_mask],
            y.loc[surface_mask],
            max_rows=5000,
        )
        calibrator.fit(surface_X, surface_y)
        surface_models[surface_col] = calibrator

    return SurfaceCalibratedModel(default_model, surface_models, SURFACE_COLUMNS), base_model


def grouped_time_series_splits(df, n_splits=4):
    if "match_id" not in df.columns:
        return TimeSeriesSplit(n_splits=n_splits)

    match_order = (
        df[["match_id", "date"]]
        .drop_duplicates("match_id")
        .sort_values(["date", "match_id"])
        ["match_id"]
        .to_list()
    )
    if len(match_order) <= n_splits:
        return TimeSeriesSplit(n_splits=n_splits)

    splits = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    match_positions = pd.Series(range(len(match_order)), index=match_order)
    for train_group_idx, test_group_idx in splitter.split(match_order):
        train_idx = df.index[df["match_id"].map(match_positions).isin(train_group_idx)]
        test_idx = df.index[df["match_id"].map(match_positions).isin(test_group_idx)]
        splits.append((train_idx.to_numpy(), test_idx.to_numpy()))

    return splits


def train_test_split_by_match(df, test_fraction=0.2):
    if "match_id" not in df.columns:
        split_idx = int(len(df) * (1 - test_fraction))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    matches = (
        df[["match_id", "date"]]
        .drop_duplicates("match_id")
        .sort_values(["date", "match_id"])
    )
    split_idx = int(len(matches) * (1 - test_fraction))
    train_matches = set(matches.iloc[:split_idx]["match_id"])
    train_df = df[df["match_id"].isin(train_matches)].copy()
    test_df = df[~df["match_id"].isin(train_matches)].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def holdout_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        "roc_auc": roc_auc_score(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),
    }


def search_model(X_train, y_train, cv_splits, search_name):
    n_iter = int(os.getenv(XGB_SEARCH_ITER_ENV, "36"))
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=cv_splits,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )

    print(f"\nSearching XGBoost params for {search_name}...")
    search.fit(X_train, y_train)
    print(f"Best CV log loss for {search_name}: {-search.best_score_:.5f}")
    print(f"Best params for {search_name}: {search.best_params_}")
    return search


def feature_importance(model, feature_cols, output_path):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_path, index=False)


def train_one_model(df, feature_cols, model_name, model_path, columns_path):
    train_df, test_df = train_test_split_by_match(df)
    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train = train_df["target"]
    y_test = test_df["target"]
    cv_splits = grouped_time_series_splits(train_df)

    saved_state = load_saved_training_state(model_name, len(feature_cols))
    search_results = pd.DataFrame()

    if (
        saved_state is None
        and os.getenv(SKIP_TUNING_ENV, "").strip().lower() in {"1", "true", "yes"}
    ):
        best_params = load_last_params(model_name)
        cv_log_loss = pd.NA
        print(f"\nSkipping XGBoost search for {model_name}; using prior/default params.")
    elif saved_state is None:
        search = search_model(X_train, y_train, cv_splits, model_name)
        best_params = search.best_params_
        cv_log_loss = -search.best_score_

        search_results = pd.DataFrame(search.cv_results_)
        search_results["model_name"] = model_name
    else:
        best_params = saved_state["best_params"]
        print(f"\nReusing saved XGBoost params for {model_name}.")
        cv_log_loss = saved_state["best_cv_log_loss"]

    validation_model, _ = fit_surface_calibrated_model(best_params, X_train, y_train)
    metrics = holdout_metrics(validation_model, X_test, y_test)

    X = df[feature_cols].fillna(0)
    y = df["target"]
    final_model, final_base_model = fit_surface_calibrated_model(best_params, X, y)

    joblib.dump(final_model, model_path)
    joblib.dump(feature_cols, columns_path)

    feature_importance(
        final_base_model,
        feature_cols,
        f"outputs/xgb_feature_importance_{model_name}.csv",
    )

    return {
        "model_name": model_name,
        "model_path": model_path,
        "feature_count": len(feature_cols),
        "best_cv_log_loss": cv_log_loss,
        "best_params": best_params,
        **metrics,
    }, search_results


def train_model():
    df = pd.read_csv(FEATURES_PATH)

    drop_cols = ["match_id", "date", "player_a", "player_b", "target"]
    all_feature_cols = [col for col in df.columns if col not in drop_cols]
    market_cols = [
        col for col in all_feature_cols
        if "market" in col or "bet365" in col or "pinnacle" in col
    ]
    pure_feature_cols = [
        col for col in all_feature_cols
        if col not in market_cols
    ]

    metrics_rows = []
    search_results = []

    pure_metrics, pure_search_results = train_one_model(
        df,
        pure_feature_cols,
        "pure_tennis",
        PURE_MODEL_PATH,
        PURE_FEATURE_COLUMNS_PATH,
    )
    metrics_rows.append(pure_metrics)
    search_results.append(pure_search_results)

    market_metrics, market_search_results = train_one_model(
        df,
        all_feature_cols,
        "market_aware",
        MARKET_MODEL_PATH,
        MARKET_FEATURE_COLUMNS_PATH,
    )
    metrics_rows.append(market_metrics)
    search_results.append(market_search_results)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_PATH, index=False)
    search_results = [result for result in search_results if not result.empty]
    if search_results:
        pd.concat(search_results, ignore_index=True).to_csv(
            PARAM_SEARCH_RESULTS_PATH,
            index=False,
        )

    best_row = metrics_df.sort_values(["log_loss", "roc_auc"], ascending=[True, False]).iloc[0]
    if best_row["model_name"] == "market_aware":
        best_model_path = MARKET_MODEL_PATH
        best_columns_path = MARKET_FEATURE_COLUMNS_PATH
        best_importance_path = "outputs/xgb_feature_importance_market_aware.csv"
    else:
        best_model_path = PURE_MODEL_PATH
        best_columns_path = PURE_FEATURE_COLUMNS_PATH
        best_importance_path = "outputs/xgb_feature_importance_pure_tennis.csv"

    joblib.dump(joblib.load(best_model_path), MODEL_PATH)
    joblib.dump(joblib.load(best_columns_path), FEATURE_COLUMNS_PATH)
    pd.read_csv(best_importance_path).to_csv(
        "outputs/xgb_feature_importance.csv",
        index=False,
    )

    print("\nHoldout metrics:")
    print(
        metrics_df[
            ["model_name", "accuracy", "log_loss", "roc_auc", "brier_score", "feature_count"]
        ]
    )
    print(f"\nDefault model saved from: {best_row['model_name']}")


if __name__ == "__main__":
    train_model()
