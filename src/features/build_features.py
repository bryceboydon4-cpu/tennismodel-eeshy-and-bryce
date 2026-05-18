import pandas as pd
from pathlib import Path

ELO_FEATURES_PATH = "data/processed/elo_features.csv"
MATCH_FEATURES_PATH = "data/processed/match_features.csv"
ODDS_DATA_GLOB = "data/odds/*.csv"


def odds_probability(win_odds, lose_odds):
    if pd.isna(win_odds) or pd.isna(lose_odds) or win_odds <= 1 or lose_odds <= 1:
        return pd.NA

    raw_win = 1 / win_odds
    raw_lose = 1 / lose_odds
    return raw_win / (raw_win + raw_lose)


def load_odds_features():
    files = sorted(Path("data/odds").glob("*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for file in files:
        odds = pd.read_csv(file)
        odds["date"] = pd.to_datetime(odds["Date"], errors="coerce")
        odds["best_of"] = pd.to_numeric(odds["Best of"], errors="coerce")
        odds["winner_rank"] = pd.to_numeric(odds["WRank"], errors="coerce")
        odds["loser_rank"] = pd.to_numeric(odds["LRank"], errors="coerce")
        odds["winner_rank_points"] = pd.to_numeric(odds["WPts"], errors="coerce")
        odds["loser_rank_points"] = pd.to_numeric(odds["LPts"], errors="coerce")

        for col in ["AvgW", "AvgL", "MaxW", "MaxL", "PSW", "PSL", "B365W", "B365L"]:
            if col in odds.columns:
                odds[col] = pd.to_numeric(odds[col], errors="coerce")

        odds["market_prob_winner"] = odds.apply(
            lambda row: odds_probability(row.get("AvgW"), row.get("AvgL")),
            axis=1,
        )
        odds["market_prob_loser"] = 1 - odds["market_prob_winner"]
        odds["max_market_prob_winner"] = odds.apply(
            lambda row: odds_probability(row.get("MaxW"), row.get("MaxL")),
            axis=1,
        )
        odds["pinnacle_prob_winner"] = odds.apply(
            lambda row: odds_probability(row.get("PSW"), row.get("PSL")),
            axis=1,
        )
        odds["bet365_prob_winner"] = odds.apply(
            lambda row: odds_probability(row.get("B365W"), row.get("B365L")),
            axis=1,
        )

        dfs.append(odds)

    odds_df = pd.concat(dfs, ignore_index=True)
    keep_cols = [
        "date",
        "Surface",
        "best_of",
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "market_prob_winner",
        "market_prob_loser",
        "max_market_prob_winner",
        "pinnacle_prob_winner",
        "bet365_prob_winner",
    ]
    odds_df = odds_df[keep_cols].rename(columns={"Surface": "surface"})
    odds_df = odds_df.dropna(subset=["date", "surface", "winner_rank", "loser_rank"])
    return odds_df.drop_duplicates(
        subset=[
            "date",
            "surface",
            "best_of",
            "winner_rank",
            "loser_rank",
            "winner_rank_points",
            "loser_rank_points",
        ],
        keep="last",
    )


def safe_diff(row, left_col, right_col):
    left = row.get(left_col)
    right = row.get(right_col)

    if pd.isna(left) or pd.isna(right):
        return 0

    return left - right


def player_view(row, player_is_winner):
    if player_is_winner:
        prefix_a = "winner"
        prefix_b = "loser"
        target = 1
        sign = 1
    else:
        prefix_a = "loser"
        prefix_b = "winner"
        target = 0
        sign = -1

    return {
        "date": row["date"],
        "surface": row["surface"],
        "tourney_level": row["tourney_level"],
        "round": row["round"],
        "best_of": row["best_of"],
        "draw_size": row["draw_size"],
        "player_a": row[prefix_a],
        "player_b": row[prefix_b],
        "elo_diff": sign * row["elo_diff"],
        "surface_elo_diff": sign * row["surface_elo_diff"],
        "rank_diff": safe_diff(row, f"{prefix_a}_rank", f"{prefix_b}_rank"),
        "rank_points_diff": safe_diff(
            row,
            f"{prefix_a}_rank_points",
            f"{prefix_b}_rank_points",
        ),
        "age_diff": safe_diff(row, f"{prefix_a}_age", f"{prefix_b}_age"),
        "market_prob_diff": sign * safe_diff(
            row,
            "market_prob_winner",
            "market_prob_loser",
        ),
        "market_prob_a": row.get(
            "market_prob_winner" if player_is_winner else "market_prob_loser",
            0,
        ),
        "max_market_prob_a": (
            row.get("max_market_prob_winner", 0)
            if player_is_winner
            else 1 - row.get("max_market_prob_winner", 0)
        ),
        "pinnacle_prob_a": (
            row.get("pinnacle_prob_winner", 0)
            if player_is_winner
            else 1 - row.get("pinnacle_prob_winner", 0)
        ),
        "bet365_prob_a": (
            row.get("bet365_prob_winner", 0)
            if player_is_winner
            else 1 - row.get("bet365_prob_winner", 0)
        ),
        "has_market_odds": int(not pd.isna(row.get("market_prob_winner"))),
        "target": target,
    }


def build_features():
    df = pd.read_csv(ELO_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    odds_df = load_odds_features()
    if not odds_df.empty:
        merge_cols = [
            "date",
            "surface",
            "best_of",
            "winner_rank",
            "loser_rank",
            "winner_rank_points",
            "loser_rank_points",
        ]
        before_count = len(df)
        df = df.merge(odds_df, on=merge_cols, how="left")
        matched_count = df["market_prob_winner"].notna().sum()
        print(f"Matched odds for {matched_count:,} of {before_count:,} matches.")

    ml_rows = []
    for _, row in df.iterrows():
        ml_rows.append(player_view(row, player_is_winner=True))
        ml_rows.append(player_view(row, player_is_winner=False))

    ml_df = pd.DataFrame(ml_rows)

    categorical_cols = ["surface", "tourney_level", "round"]
    ml_df = pd.get_dummies(
        ml_df,
        columns=categorical_cols,
        dummy_na=True,
        dtype=int,
    )

    ml_df.to_csv(MATCH_FEATURES_PATH, index=False)
    print("Feature dataset created!")


if __name__ == "__main__":
    build_features()
