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
PLAYER_FORM_PATH = OUTPUT_DIR / "current_player_form.csv"
ROLLING_STAT_NAMES = (
    "last_5_win_rate",
    "last_10_win_rate",
    "surface_last_10_win_rate",
    "days_since_last_match",
    "matches_last_14_days",
    "ace_rate",
    "double_fault_rate",
    "first_serve_points_won",
    "second_serve_points_won",
    "return_points_won",
    "break_points_saved",
    "break_points_converted",
    "last_10_ace_rate",
    "last_10_double_fault_rate",
    "last_10_first_serve_points_won",
    "last_10_second_serve_points_won",
    "last_10_return_points_won",
    "last_10_break_points_saved",
    "last_10_break_points_converted",
    "surface_last_10_ace_rate",
    "surface_last_10_double_fault_rate",
    "surface_last_10_first_serve_points_won",
    "surface_last_10_second_serve_points_won",
    "surface_last_10_return_points_won",
    "surface_last_10_break_points_saved",
    "surface_last_10_break_points_converted",
    "matches_last_3_days",
    "minutes_last_7_days",
    "sets_last_7_days",
    "retired_last_match",
    "retirements_last_90_days",
    "walkovers_last_90_days",
)
RATE_STAT_NAMES = (
    "ace_rate",
    "double_fault_rate",
    "first_serve_points_won",
    "second_serve_points_won",
    "return_points_won",
    "break_points_saved",
    "break_points_converted",
)

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


def safe_number(value, default=0):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return numeric


def rate(numerator, denominator):
    if denominator <= 0:
        return 0
    return numerator / denominator


def recent_win_rate(results, window):
    recent = results[-window:]
    if not recent:
        return 0
    return sum(recent) / len(recent)


def recent_match_count(dates, current_date, days):
    cutoff = current_date - pd.Timedelta(days=days)
    return sum(cutoff <= match_date < current_date for match_date in dates)


def recent_sum(values, dates, current_date, days):
    cutoff = current_date - pd.Timedelta(days=days)
    return sum(
        value
        for value, item_date in zip(values, dates)
        if cutoff <= item_date < current_date
    )


def recent_event_count(dates, current_date, days):
    cutoff = current_date - pd.Timedelta(days=days)
    return sum(cutoff <= item_date < current_date for item_date in dates)


def recent_average(values, window):
    recent = values[-window:]
    if not recent:
        return 0
    return sum(recent) / len(recent)


def score_text(row):
    score = row.get("score")
    if pd.isna(score):
        return ""
    return str(score).upper()


def is_walkover(row):
    score = score_text(row)
    return "W/O" in score or "WO" == score.strip() or "WALKOVER" in score


def is_retirement(row):
    score = score_text(row)
    return "RET" in score or "ABN" in score or "DEF" in score


def count_sets(score):
    if pd.isna(score):
        return 0

    count = 0
    for token in str(score).replace(",", " ").split():
        token = token.upper()
        if "-" in token and not any(marker in token for marker in ("RET", "W/O", "WO")):
            count += 1
    return count


def dynamic_k_factor(row):
    level = row.get("tourney_level")
    round_name = row.get("round")
    best_of = safe_number(row.get("best_of"), default=3)
    base_by_level = {
        "G": 40,
        "F": 38,
        "M": 36,
        "A": 32,
        "D": 28,
    }
    k = base_by_level.get(level, 30)

    if round_name in {"F", "SF"}:
        k += 4
    elif round_name in {"QF", "R16"}:
        k += 2

    if best_of >= 5:
        k += 2

    return k


def snapshot_player_form(
    player,
    surface,
    current_date,
    match_results,
    surface_match_results,
    match_dates,
    stat_totals,
    stat_rate_history,
    surface_stat_rate_history,
    minutes_history,
    sets_history,
    retired_match_history,
    retirement_dates,
    walkover_dates,
):
    totals = stat_totals[player]
    previous_dates = match_dates[player]
    days_since_last_match = (
        (current_date - previous_dates[-1]).days
        if previous_dates
        else 365
    )

    form = {
        "last_5_win_rate": recent_win_rate(match_results[player], 5),
        "last_10_win_rate": recent_win_rate(match_results[player], 10),
        "surface_last_10_win_rate": recent_win_rate(
            surface_match_results[(player, surface)],
            10,
        ),
        "days_since_last_match": days_since_last_match,
        "matches_last_14_days": recent_match_count(previous_dates, current_date, 14),
        "matches_last_3_days": recent_match_count(previous_dates, current_date, 3),
        "minutes_last_7_days": recent_sum(
            minutes_history[player],
            previous_dates,
            current_date,
            7,
        ),
        "sets_last_7_days": recent_sum(
            sets_history[player],
            previous_dates,
            current_date,
            7,
        ),
        "retired_last_match": (
            int(retired_match_history[player][-1])
            if retired_match_history[player]
            else 0
        ),
        "retirements_last_90_days": recent_event_count(
            retirement_dates[player],
            current_date,
            90,
        ),
        "walkovers_last_90_days": recent_event_count(
            walkover_dates[player],
            current_date,
            90,
        ),
        "ace_rate": rate(totals["aces"], totals["service_points"]),
        "double_fault_rate": rate(totals["double_faults"], totals["service_points"]),
        "first_serve_points_won": rate(
            totals["first_serve_won"],
            totals["first_serve_in"],
        ),
        "second_serve_points_won": rate(
            totals["second_serve_won"],
            totals["second_serve_points"],
        ),
        "return_points_won": rate(
            totals["return_points_won"],
            totals["return_points"],
        ),
        "break_points_saved": rate(
            totals["break_points_saved"],
            totals["break_points_faced"],
        ),
        "break_points_converted": rate(
            totals["break_points_converted"],
            totals["break_points_return"],
        ),
    }
    for stat_name in RATE_STAT_NAMES:
        form[f"last_10_{stat_name}"] = recent_average(
            stat_rate_history[(player, stat_name)],
            10,
        )
        form[f"surface_last_10_{stat_name}"] = recent_average(
            surface_stat_rate_history[(player, surface, stat_name)],
            10,
        )
    return form


def add_player_form(row, prefix, opponent_prefix, totals):
    service_points = safe_number(row.get(f"{prefix}_svpt"))
    first_serve_in = safe_number(row.get(f"{prefix}_1stIn"))
    first_serve_won = safe_number(row.get(f"{prefix}_1stWon"))
    second_serve_won = safe_number(row.get(f"{prefix}_2ndWon"))
    break_points_faced = safe_number(row.get(f"{prefix}_bpFaced"))
    break_points_saved = safe_number(row.get(f"{prefix}_bpSaved"))

    opponent_service_points = safe_number(row.get(f"{opponent_prefix}_svpt"))
    opponent_first_serve_won = safe_number(row.get(f"{opponent_prefix}_1stWon"))
    opponent_second_serve_won = safe_number(row.get(f"{opponent_prefix}_2ndWon"))
    opponent_break_points_faced = safe_number(row.get(f"{opponent_prefix}_bpFaced"))
    opponent_break_points_saved = safe_number(row.get(f"{opponent_prefix}_bpSaved"))

    second_serve_points = max(service_points - first_serve_in, 0)
    return_points_won = max(
        opponent_service_points - opponent_first_serve_won - opponent_second_serve_won,
        0,
    )
    break_points_converted = max(
        opponent_break_points_faced - opponent_break_points_saved,
        0,
    )

    totals["aces"] += safe_number(row.get(f"{prefix}_ace"))
    totals["double_faults"] += safe_number(row.get(f"{prefix}_df"))
    totals["service_points"] += service_points
    totals["first_serve_in"] += first_serve_in
    totals["first_serve_won"] += first_serve_won
    totals["second_serve_points"] += second_serve_points
    totals["second_serve_won"] += second_serve_won
    totals["return_points"] += opponent_service_points
    totals["return_points_won"] += return_points_won
    totals["break_points_faced"] += break_points_faced
    totals["break_points_saved"] += break_points_saved
    totals["break_points_return"] += opponent_break_points_faced
    totals["break_points_converted"] += break_points_converted

    return {
        "ace_rate": rate(safe_number(row.get(f"{prefix}_ace")), service_points),
        "double_fault_rate": rate(safe_number(row.get(f"{prefix}_df")), service_points),
        "first_serve_points_won": rate(first_serve_won, first_serve_in),
        "second_serve_points_won": rate(second_serve_won, second_serve_points),
        "return_points_won": rate(return_points_won, opponent_service_points),
        "break_points_saved": rate(break_points_saved, break_points_faced),
        "break_points_converted": rate(
            break_points_converted,
            opponent_break_points_faced,
        ),
    }


def make_player_form_table(
    overall_elos,
    match_results,
    surface_match_results,
    match_dates,
    stat_totals,
    stat_rate_history,
    surface_stat_rate_history,
    minutes_history,
    sets_history,
    retired_match_history,
    retirement_dates,
    walkover_dates,
    as_of_date,
):
    rows = []
    for player in sorted(overall_elos.keys()):
        row = {"player": player}
        for surface in SURFACES:
            form = snapshot_player_form(
                player,
                surface,
                as_of_date,
                match_results,
                surface_match_results,
                match_dates,
                stat_totals,
                stat_rate_history,
                surface_stat_rate_history,
                minutes_history,
                sets_history,
                retired_match_history,
                retirement_dates,
                walkover_dates,
            )
            for stat_name, value in form.items():
                row[f"{surface.lower()}_{stat_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def h2h_key(player_a, player_b):
    return tuple(sorted((player_a, player_b)))


def h2h_snapshot(player_a, player_b, surface, h2h_results, surface_h2h_results):
    pair_key = h2h_key(player_a, player_b)
    all_results = h2h_results[pair_key]
    surface_results = surface_h2h_results[(pair_key, surface)]

    return {
        "h2h_meetings": len(all_results),
        "h2h_win_rate": recent_win_rate(
            [int(winner == player_a) for winner in all_results],
            len(all_results),
        ),
        "surface_h2h_meetings": len(surface_results),
        "surface_h2h_win_rate": recent_win_rate(
            [int(winner == player_a) for winner in surface_results],
            len(surface_results),
        ),
        "last_h2h_win": int(all_results[-1] == player_a) if all_results else 0,
    }


def process_matches():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    overall_elos = defaultdict(lambda: STARTING_ELO)
    surface_elos = {
        surface: defaultdict(lambda: STARTING_ELO)
        for surface in SURFACES
    }
    match_results = defaultdict(list)
    surface_match_results = defaultdict(list)
    match_dates = defaultdict(list)
    stat_totals = defaultdict(lambda: defaultdict(float))
    stat_rate_history = defaultdict(list)
    surface_stat_rate_history = defaultdict(list)
    minutes_history = defaultdict(list)
    sets_history = defaultdict(list)
    retired_match_history = defaultdict(list)
    retirement_dates = defaultdict(list)
    walkover_dates = defaultdict(list)
    h2h_results = defaultdict(list)
    surface_h2h_results = defaultdict(list)

    match_rows = []

    for _, row in matches.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row.get("surface")
        current_date = row["tourney_date"]

        if pd.isna(winner) or pd.isna(loser) or surface not in surface_elos:
            continue

        if is_walkover(row):
            walkover_dates[winner].append(current_date)
            walkover_dates[loser].append(current_date)
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
        winner_form = snapshot_player_form(
            winner,
            surface,
            current_date,
            match_results,
            surface_match_results,
            match_dates,
            stat_totals,
            stat_rate_history,
            surface_stat_rate_history,
            minutes_history,
            sets_history,
            retired_match_history,
            retirement_dates,
            walkover_dates,
        )
        loser_form = snapshot_player_form(
            loser,
            surface,
            current_date,
            match_results,
            surface_match_results,
            match_dates,
            stat_totals,
            stat_rate_history,
            surface_stat_rate_history,
            minutes_history,
            sets_history,
            retired_match_history,
            retirement_dates,
            walkover_dates,
        )
        winner_h2h = h2h_snapshot(
            winner,
            loser,
            surface,
            h2h_results,
            surface_h2h_results,
        )
        loser_h2h = h2h_snapshot(
            loser,
            winner,
            surface,
            h2h_results,
            surface_h2h_results,
        )

        match_row = {
            "match_id": f"{row.get('tourney_id')}_{row.get('match_num')}",
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
            "winner_hand": row.get("winner_hand"),
            "loser_hand": row.get("loser_hand"),
            "winner_ht": row.get("winner_ht"),
            "loser_ht": row.get("loser_ht"),
            "winner_ioc": row.get("winner_ioc"),
            "loser_ioc": row.get("loser_ioc"),
            "winner_seed": row.get("winner_seed"),
            "loser_seed": row.get("loser_seed"),
            "winner_entry": row.get("winner_entry"),
            "loser_entry": row.get("loser_entry"),
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
        }
        for stat_name in ROLLING_STAT_NAMES:
            match_row[f"winner_{stat_name}"] = winner_form[stat_name]
            match_row[f"loser_{stat_name}"] = loser_form[stat_name]
        for stat_name, value in winner_h2h.items():
            match_row[f"winner_{stat_name}"] = value
        for stat_name, value in loser_h2h.items():
            match_row[f"loser_{stat_name}"] = value
        match_rows.append(match_row)

        k = dynamic_k_factor(row)
        overall_elos[winner] = update_elo(winner_elo, winner_expected, 1, k=k)
        overall_elos[loser] = update_elo(loser_elo, loser_expected, 0, k=k)
        surface_elos[surface][winner] = update_elo(
            winner_surface_elo,
            winner_surface_expected,
            1,
            k=k,
        )
        surface_elos[surface][loser] = update_elo(
            loser_surface_elo,
            loser_surface_expected,
            0,
            k=k,
        )
        match_results[winner].append(1)
        match_results[loser].append(0)
        surface_match_results[(winner, surface)].append(1)
        surface_match_results[(loser, surface)].append(0)
        match_dates[winner].append(current_date)
        match_dates[loser].append(current_date)
        minutes = safe_number(row.get("minutes"))
        sets_played = count_sets(row.get("score"))
        minutes_history[winner].append(minutes)
        minutes_history[loser].append(minutes)
        sets_history[winner].append(sets_played)
        sets_history[loser].append(sets_played)
        loser_retired = is_retirement(row)
        retired_match_history[winner].append(0)
        retired_match_history[loser].append(int(loser_retired))
        if loser_retired:
            retirement_dates[loser].append(current_date)
        winner_rates = add_player_form(row, "w", "l", stat_totals[winner])
        loser_rates = add_player_form(row, "l", "w", stat_totals[loser])
        for stat_name, value in winner_rates.items():
            stat_rate_history[(winner, stat_name)].append(value)
            surface_stat_rate_history[(winner, surface, stat_name)].append(value)
        for stat_name, value in loser_rates.items():
            stat_rate_history[(loser, stat_name)].append(value)
            surface_stat_rate_history[(loser, surface, stat_name)].append(value)
        pair_key = h2h_key(winner, loser)
        h2h_results[pair_key].append(winner)
        surface_h2h_results[(pair_key, surface)].append(winner)

    as_of_date = matches["tourney_date"].max() + pd.Timedelta(days=1)
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

    make_player_form_table(
        overall_elos,
        match_results,
        surface_match_results,
        match_dates,
        stat_totals,
        stat_rate_history,
        surface_stat_rate_history,
        minutes_history,
        sets_history,
        retired_match_history,
        retirement_dates,
        walkover_dates,
        as_of_date,
    ).to_csv(PLAYER_FORM_PATH, index=False)

    print("\nTOP 20 OVERALL ELO PLAYERS\n")
    for _, row in rankings.head(20).iterrows():
        print(f"{row['player']}: {row['overall_elo']:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    process_matches()
