import argparse

import joblib
import pandas as pd


FEATURES_PATH = "data/processed/match_features.csv"
MODEL_PATH = "outputs/xgb_model.pkl"
FEATURE_COLUMNS_PATH = "outputs/xgb_feature_columns.pkl"


def split_recent_holdout(df, holdout_fraction):
    matches = (
        df[["match_id", "date"]]
        .drop_duplicates("match_id")
        .sort_values(["date", "match_id"])
    )
    split_idx = int(len(matches) * (1 - holdout_fraction))
    holdout_matches = set(matches.iloc[split_idx:]["match_id"])
    return df[df["match_id"].isin(holdout_matches)].copy()


def backtest(edge_threshold=0.03, min_odds=1.01, max_odds=20.0, holdout_fraction=0.2):
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = split_recent_holdout(df, holdout_fraction)

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    X = df.reindex(columns=feature_columns, fill_value=0).fillna(0)

    df["model_prob"] = model.predict_proba(X)[:, 1]
    df["edge"] = df["model_prob"] - df["market_prob_a"]
    df["expected_value"] = df["model_prob"] * df["market_odds_a"] - 1

    bets = df[
        (df["has_market_odds"] == 1)
        & (df["market_odds_a"] >= min_odds)
        & (df["market_odds_a"] <= max_odds)
        & (df["edge"] >= edge_threshold)
    ].copy()

    bets["profit"] = bets.apply(
        lambda row: row["market_odds_a"] - 1 if row["target"] == 1 else -1,
        axis=1,
    )

    stake = len(bets)
    profit = bets["profit"].sum() if stake else 0
    roi = profit / stake if stake else 0
    win_rate = bets["target"].mean() if stake else 0

    return {
        "holdout_start": df["date"].min(),
        "holdout_end": df["date"].max(),
        "edge_threshold": edge_threshold,
        "min_odds": min_odds,
        "max_odds": max_odds,
        "bets": stake,
        "wins": int(bets["target"].sum()) if stake else 0,
        "win_rate": win_rate,
        "profit_units": profit,
        "roi": roi,
        "avg_edge": bets["edge"].mean() if stake else 0,
        "avg_expected_value": bets["expected_value"].mean() if stake else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest model edge/EV bets on the most recent holdout period.",
    )
    parser.add_argument("--edge", type=float, default=0.03)
    parser.add_argument("--min-odds", type=float, default=1.01)
    parser.add_argument("--max-odds", type=float, default=20.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    result = backtest(
        edge_threshold=args.edge,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        holdout_fraction=args.holdout_fraction,
    )

    print(f"Holdout: {result['holdout_start'].date()} to {result['holdout_end'].date()}")
    print(f"Edge threshold: {result['edge_threshold']:.1%}")
    print(f"Odds range: {result['min_odds']:.2f} to {result['max_odds']:.2f}")
    print(f"Bets: {result['bets']}")
    print(f"Wins: {result['wins']}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Profit: {result['profit_units']:+.2f} units")
    print(f"ROI: {result['roi']:+.2%}")
    print(f"Average edge: {result['avg_edge']:.2%}")
    print(f"Average EV: {result['avg_expected_value']:.2%}")


if __name__ == "__main__":
    main()
