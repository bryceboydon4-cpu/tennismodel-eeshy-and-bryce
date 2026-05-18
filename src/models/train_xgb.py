import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
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


def holdout_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def search_model(X_train, y_train, search_name):
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    time_splits = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRID,
        n_iter=36,
        scoring="neg_log_loss",
        cv=time_splits,
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
    X = df[feature_cols].fillna(0)
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    search = search_model(X_train, y_train, model_name)
    metrics = holdout_metrics(search.best_estimator_, X_test, y_test)

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **search.best_params_,
    )
    final_model.fit(X, y)

    joblib.dump(final_model, model_path)
    joblib.dump(feature_cols, columns_path)

    feature_importance(
        final_model,
        feature_cols,
        f"outputs/xgb_feature_importance_{model_name}.csv",
    )

    search_results = pd.DataFrame(search.cv_results_)
    search_results["model_name"] = model_name

    return {
        "model_name": model_name,
        "model_path": model_path,
        "feature_count": len(feature_cols),
        "best_cv_log_loss": -search.best_score_,
        "best_params": search.best_params_,
        **metrics,
    }, search_results


def train_model():
    df = pd.read_csv(FEATURES_PATH)

    drop_cols = ["date", "player_a", "player_b", "target"]
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
    print(metrics_df[["model_name", "accuracy", "log_loss", "roc_auc", "feature_count"]])
    print(f"\nDefault model saved from: {best_row['model_name']}")


if __name__ == "__main__":
    train_model()
