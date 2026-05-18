from collections import defaultdict
import glob
from pathlib import Path

import pandas as pd

from src.elo.elo import expected_score, update_elo


RAW_DATA_GLOB = "data/raw/atp_matches*.csv"
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")
STARTING_ELO = 1500
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def load_matches(raw_data_glob=RAW_DATA_GLOB):
    files = sorted(glob.glob(raw_data_glob))
    if not files:
        raise FileNotFoundError(f"No raw match files found at {raw_data_glob}")

    dfs = []
    for file in files:
        print(f"Loading: {file}")
        dfs.append(pd.read_csv(file))

    matches = pd.concat(dfs, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"],
        format="%Y%m%d",
        errors="coerce",
    )

    matches = matches.dropna(subset=["tourney_date"])
    matches = matches.sort_values(["tourney_date", "tourney_id", "match_num"])
    return matches


def make_rankings(overall_elos, surface_elos):
    players = sorted(overall_elos.keys())
    rows = []

    for player in players:
        row = {
            "player": player,
            "overall_elo": overall_elos[player],
        }
        for surface in SURFACES:
            row[f"{surface.lower()}_elo"] = surface_elos[surface].get(
                player,
                STARTING_ELO,
            )
        rows.append(row)

    rankings = pd.DataFrame(rows)
    return rankings.sort_values("overall_elo", ascending=False)


def process_matches():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    overall_elos = defaultdict(lambda: STARTING_ELO)
    surface_elos = {
        surface: defaultdict(lambda: STARTING_ELO)
        for surface in SURFACES
    }

    match_rows = []

    for _, row in matches.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row.get("surface")

        if pd.isna(winner) or pd.isna(loser) or surface not in surface_elos:
            continue

        winner_elo = overall_elos[winner]
        loser_elo = overall_elos[loser]
        winner_surface_elo = surface_elos[surface][winner]
        loser_surface_elo = surface_elos[surface][loser]

        winner_expected = expected_score(winner_elo, loser_elo)
        loser_expected = expected_score(loser_elo, winner_elo)
        winner_surface_expected = expected_score(
            winner_surface_elo,
            loser_surface_elo,
        )
        loser_surface_expected = expected_score(
            loser_surface_elo,
            winner_surface_elo,
        )

        match_rows.append({
            "date": row["tourney_date"],
            "tourney_id": row.get("tourney_id"),
            "tourney_name": row.get("tourney_name"),
            "surface": surface,
            "tourney_level": row.get("tourney_level"),
            "round": row.get("round"),
            "best_of": row.get("best_of"),
            "draw_size": row.get("draw_size"),
            "winner": winner,
            "loser": loser,
            "winner_age": row.get("winner_age"),
            "loser_age": row.get("loser_age"),
            "winner_rank": row.get("winner_rank"),
            "loser_rank": row.get("loser_rank"),
            "winner_rank_points": row.get("winner_rank_points"),
            "loser_rank_points": row.get("loser_rank_points"),
            "winner_elo": winner_elo,
            "loser_elo": loser_elo,
            "winner_surface_elo": winner_surface_elo,
            "loser_surface_elo": loser_surface_elo,
            "elo_diff": winner_elo - loser_elo,
            "surface_elo_diff": winner_surface_elo - loser_surface_elo,
            "winner_expected": winner_expected,
            "winner_surface_expected": winner_surface_expected,
            "target": 1,
        })

        overall_elos[winner] = update_elo(winner_elo, winner_expected, 1)
        overall_elos[loser] = update_elo(loser_elo, loser_expected, 0)
        surface_elos[surface][winner] = update_elo(
            winner_surface_elo,
            winner_surface_expected,
            1,
        )
        surface_elos[surface][loser] = update_elo(
            loser_surface_elo,
            loser_surface_expected,
            0,
        )

    feature_df = pd.DataFrame(match_rows)
    feature_df.to_csv(PROCESSED_DIR / "elo_features.csv", index=False)

    rankings = make_rankings(overall_elos, surface_elos)
    rankings.to_csv(OUTPUT_DIR / "current_elo_rankings.csv", index=False)

    surface_rankings = []
    for surface, ratings in surface_elos.items():
        for player, elo in ratings.items():
            surface_rankings.append({
                "surface": surface,
                "player": player,
                "surface_elo": elo,
            })

    pd.DataFrame(surface_rankings).sort_values(
        ["surface", "surface_elo"],
        ascending=[True, False],
    ).to_csv(OUTPUT_DIR / "current_surface_elo_rankings.csv", index=False)

    print("\nTOP 20 OVERALL ELO PLAYERS\n")
    for _, row in rankings.head(20).iterrows():
        print(f"{row['player']}: {row['overall_elo']:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    process_matches()
