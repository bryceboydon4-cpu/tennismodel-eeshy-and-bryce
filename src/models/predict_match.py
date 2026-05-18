import argparse

import joblib
import pandas as pd


MODEL_PATH = "outputs/xgb_model.pkl"
FEATURE_COLUMNS_PATH = "outputs/xgb_feature_columns.pkl"
RATINGS_PATH = "outputs/current_elo_rankings.csv"
SURFACE_NAMES = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


def load_ratings():
    ratings = pd.read_csv(RATINGS_PATH)
    return ratings.set_index("player")


def normalize_player_name(name):
    return " ".join(name.split())


def resolve_player_name(name, ratings):
    normalized = normalize_player_name(name)
    if normalized in ratings.index:
        return normalized

    lookup = {
        normalize_player_name(player).casefold(): player
        for player in ratings.index
    }
    resolved = lookup.get(normalized.casefold())
    if resolved is not None:
        return resolved

    matches = [
        player for player in ratings.index
        if normalized.casefold() in normalize_player_name(player).casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sample = ", ".join(matches[:8])
        raise ValueError(
            f"Player name {name!r} is ambiguous. Possible matches: {sample}"
        )

    raise ValueError(f"Could not find ratings for {normalized}")


def normalize_surface(surface):
    normalized = surface.strip().casefold()
    if normalized not in SURFACE_NAMES:
        valid = ", ".join(SURFACE_NAMES.values())
        raise ValueError(f"Unknown surface: {surface}. Use one of: {valid}")
    return SURFACE_NAMES[normalized]


def surface_column(surface):
    return f"{surface.lower()}_elo"


def build_prediction_row(
    player_a_elo,
    player_b_elo,
    player_a_surface_elo,
    player_b_surface_elo,
    surface,
    best_of=3,
    draw_size=32,
    tourney_level=None,
    round_name=None,
    player_a_rank=None,
    player_b_rank=None,
    player_a_rank_points=None,
    player_b_rank_points=None,
    player_a_age=None,
    player_b_age=None,
    player_a_market_odds=None,
    player_b_market_odds=None,
):
    market_prob_a = 0
    market_prob_b = 0
    has_market_odds = 0
    if (
        player_a_market_odds is not None
        and player_b_market_odds is not None
        and player_a_market_odds > 1
        and player_b_market_odds > 1
    ):
        raw_a = 1 / player_a_market_odds
        raw_b = 1 / player_b_market_odds
        market_prob_a = raw_a / (raw_a + raw_b)
        market_prob_b = raw_b / (raw_a + raw_b)
        has_market_odds = 1

    row = {
        "elo_diff": player_a_elo - player_b_elo,
        "surface_elo_diff": player_a_surface_elo - player_b_surface_elo,
        "rank_diff": 0 if player_a_rank is None or player_b_rank is None else player_a_rank - player_b_rank,
        "rank_points_diff": 0 if player_a_rank_points is None or player_b_rank_points is None else player_a_rank_points - player_b_rank_points,
        "age_diff": 0 if player_a_age is None or player_b_age is None else player_a_age - player_b_age,
        "market_prob_diff": market_prob_a - market_prob_b,
        "market_prob_a": market_prob_a,
        "max_market_prob_a": market_prob_a,
        "pinnacle_prob_a": market_prob_a,
        "bet365_prob_a": market_prob_a,
        "has_market_odds": has_market_odds,
        "best_of": best_of,
        "draw_size": draw_size,
    }

    row[f"surface_{surface}"] = 1
    if tourney_level is not None:
        row[f"tourney_level_{tourney_level}"] = 1
    if round_name is not None:
        row[f"round_{round_name}"] = 1

    prediction_df = pd.DataFrame([row])
    prediction_df = prediction_df.reindex(columns=feature_columns, fill_value=0)
    return prediction_df


def predict_match_from_elos(
    player_a_elo,
    player_b_elo,
    player_a_surface_elo,
    player_b_surface_elo,
    surface,
    **context,
):
    prediction_df = build_prediction_row(
        player_a_elo,
        player_b_elo,
        player_a_surface_elo,
        player_b_surface_elo,
        surface,
        **context,
    )
    prob_a = model.predict_proba(prediction_df)[0][1]

    return {
        "player_a_win_prob": float(prob_a),
        "player_b_win_prob": float(1 - prob_a),
    }


def predict_match(player_a, player_b, surface, **context):
    ratings = load_ratings()
    player_a = resolve_player_name(player_a, ratings)
    player_b = resolve_player_name(player_b, ratings)
    surface = normalize_surface(surface)
    surface_elo_col = surface_column(surface)

    if surface_elo_col not in ratings.columns:
        raise ValueError(f"Unknown surface: {surface}")

    result = predict_match_from_elos(
        ratings.loc[player_a, "overall_elo"],
        ratings.loc[player_b, "overall_elo"],
        ratings.loc[player_a, surface_elo_col],
        ratings.loc[player_b, surface_elo_col],
        surface,
        **context,
    )
    result["player_a"] = player_a
    result["player_b"] = player_b
    result["surface"] = surface
    result["player_a_fair_decimal_odds"] = (
        1 / result["player_a_win_prob"]
        if result["player_a_win_prob"] > 0
        else None
    )
    result["player_b_fair_decimal_odds"] = (
        1 / result["player_b_win_prob"]
        if result["player_b_win_prob"] > 0
        else None
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a tennis matchup from current Elo/XGBoost outputs.",
    )
    parser.add_argument("player_a")
    parser.add_argument("player_b")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--tourney-level", default=None)
    parser.add_argument("--round", dest="round_name", default=None)
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--draw-size", type=int, default=32)
    parser.add_argument("--player-a-rank", type=float, default=None)
    parser.add_argument("--player-b-rank", type=float, default=None)
    parser.add_argument("--player-a-rank-points", type=float, default=None)
    parser.add_argument("--player-b-rank-points", type=float, default=None)
    parser.add_argument("--player-a-age", type=float, default=None)
    parser.add_argument("--player-b-age", type=float, default=None)
    parser.add_argument("--player-a-market-odds", type=float, default=None)
    parser.add_argument("--player-b-market-odds", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_match(
        args.player_a,
        args.player_b,
        args.surface,
        tourney_level=args.tourney_level,
        round_name=args.round_name,
        best_of=args.best_of,
        draw_size=args.draw_size,
        player_a_rank=args.player_a_rank,
        player_b_rank=args.player_b_rank,
        player_a_rank_points=args.player_a_rank_points,
        player_b_rank_points=args.player_b_rank_points,
        player_a_age=args.player_a_age,
        player_b_age=args.player_b_age,
        player_a_market_odds=args.player_a_market_odds,
        player_b_market_odds=args.player_b_market_odds,
    )

    print(f"{result['player_a']} win probability: {result['player_a_win_prob']:.3%}")
    print(f"{result['player_b']} win probability: {result['player_b_win_prob']:.3%}")
    print(f"{result['player_a']} fair decimal odds: {result['player_a_fair_decimal_odds']:.2f}")
    print(f"{result['player_b']} fair decimal odds: {result['player_b_fair_decimal_odds']:.2f}")


if __name__ == "__main__":
    main()
