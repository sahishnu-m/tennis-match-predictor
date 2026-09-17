"""
Phase 3: build the table the models learn from.

For every match we describe BOTH players using only what was known before
the match started:
  - Elo ratings (Phase 2)
  - serve / return stats from their last 20 matches (serve_stats.py)
  - fatigue:  minutes and matches played in the previous 14 days
  - rust:     days since their last match
  - injury:   retirements in the previous 90 days
  - form:     wins in their last 10 matches
  - head-to-head record against this opponent
  - ranking and age (published before the tournament, so safe to use)

Then we turn each pair into DIFFERENCES ("Player A minus Player B").

WHY RANDOM PLAYER A / PLAYER B?
The raw data always lists the winner first. If Player A were always the
winner, the answer would always be "A wins" and the model would learn nothing.
So we flip a (seeded, repeatable) coin for each match to decide who is "A".

Run it:
    python -m src.features
"""

import numpy as np
import pandas as pd

from src.data_loader import APP_DATA_DIR, PROCESSED_DIR, load_matches
from src.elo import add_elo_ratings, blended, win_probability
from src.serve_stats import STAT_NAMES, add_serve_stats

RANDOM_SEED = 42
MODEL_TABLE_FILE = PROCESSED_DIR / "model_table.csv"
PLAYERS_FILE = APP_DATA_DIR / "players.csv"   # every player's latest numbers (for the app)
H2H_FILE = APP_DATA_DIR / "h2h.csv"           # head-to-head records (for the app)

MAX_REST_DAYS = 365   # "rust" is capped at one year (also used for a player's first match)


# ---------------------------------------------------------------------------
# Step 1: fill in missing match lengths
# ---------------------------------------------------------------------------

def fill_minutes(matches: pd.DataFrame) -> pd.Series:
    """
    About 1 in 5 matches has no recorded length. We fill each gap with the
    average length of other matches at the same tournament. If the whole
    tournament is missing, we use the average for that level of event and
    match format (best of 3 or best of 5).
    """
    minutes = matches["minutes"].where(matches["minutes"].between(10, 700))  # drop typos
    by_tournament = minutes.groupby(matches["tourney_id"]).transform("mean")
    by_level = minutes.groupby([matches["tourney_level"], matches["best_of"]]).transform("mean")
    by_format = minutes.groupby(matches["best_of"]).transform("mean")
    return minutes.fillna(by_tournament).fillna(by_level).fillna(by_format)


# ---------------------------------------------------------------------------
# Step 2: per-player history features
# ---------------------------------------------------------------------------

def player_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match (winner's view + loser's view), in time order."""
    common = {
        "match_id": matches["match_id"],
        "date": matches["tourney_date"],
        "minutes": fill_minutes(matches),
    }
    winners = pd.DataFrame({**common, "player_id": matches["winner_id"], "won": 1,
                            "retired": 0})
    # In a retirement, the player who quit is always listed as the loser.
    losers = pd.DataFrame({**common, "player_id": matches["loser_id"], "won": 0,
                           "retired": matches["is_retirement"].astype(int)})
    rows = pd.concat([winners, losers], ignore_index=True)
    return rows.sort_values(["player_id", "match_id"], kind="mergesort").reset_index(drop=True)


def sum_in_last_days(rows: pd.DataFrame, column: str, days: int) -> np.ndarray:
    """
    For each row: the total of `column` over the same player's EARLIER rows
    whose date is within the last `days` days.

    Earlier rounds of the same tournament count (they share a date but come
    first), which matters for fatigue: a 5-hour quarter-final tires you out
    for the semi-final.

    How it works: rows are sorted by player, then time. We give each row a
    number that goes up steadily (player number * big + day number). Then
    "the first row inside the window" is found with a fast binary search,
    and the window total is (running total now) - (running total at that row).
    """
    player_code = pd.factorize(rows["player_id"])[0].astype(np.int64)
    day = (rows["date"] - pd.Timestamp("1990-01-01")).dt.days.to_numpy().astype(np.int64)
    key = player_code * 1_000_000 + day

    running = np.concatenate([[0.0], np.cumsum(rows[column].to_numpy(dtype=float))])
    window_start = np.searchsorted(key, key - days, side="left")
    here = np.arange(len(rows))
    return running[here] - running[window_start]


def add_history_features(rows: pd.DataFrame) -> pd.DataFrame:
    g = rows.groupby("player_id")

    # Rust: days since the player's previous match.
    rest = (rows["date"] - g["date"].shift(1)).dt.days
    rows["days_since_last"] = rest.fillna(MAX_REST_DAYS).clip(upper=MAX_REST_DAYS)

    # Fatigue: time on court and number of matches in the last two weeks.
    rows["one"] = 1
    rows["minutes_14d"] = sum_in_last_days(rows, "minutes", 14)
    rows["matches_14d"] = sum_in_last_days(rows, "one", 14)

    # Injury proxy: how many times the player quit a match in the last 90 days.
    rows["retirements_90d"] = sum_in_last_days(rows, "retired", 90)

    # Form: wins in the last 10 matches (shift(1) = don't count this match).
    rows["wins_last10"] = g["won"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()).fillna(0)

    # Experience: matches played before this one.
    rows["career_matches"] = g.cumcount()
    return rows.drop(columns="one")


HISTORY_FEATURES = ["days_since_last", "minutes_14d", "matches_14d",
                    "retirements_90d", "wins_last10"]


def add_head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    """Wins each player had against this exact opponent before this match."""
    lo = np.minimum(matches["winner_id"], matches["loser_id"])
    hi = np.maximum(matches["winner_id"], matches["loser_id"])
    # Label the two players by id so every meeting of the same pair looks alike.
    lo_won = (matches["winner_id"] == lo).astype(int)
    hi_won = 1 - lo_won

    # Running count of wins for each of the pair, minus this match's result.
    lo_wins_before = lo_won.groupby([lo, hi]).cumsum() - lo_won
    hi_wins_before = hi_won.groupby([lo, hi]).cumsum() - hi_won

    out = pd.DataFrame(index=matches.index)
    winner_is_lo = matches["winner_id"] == lo
    out["winner_h2h_wins"] = np.where(winner_is_lo, lo_wins_before, hi_wins_before)
    out["loser_h2h_wins"] = np.where(winner_is_lo, hi_wins_before, lo_wins_before)
    return out


# ---------------------------------------------------------------------------
# Step 3: put everything together and pick Player A / Player B
# ---------------------------------------------------------------------------

def build_player_side_table(matches: pd.DataFrame):
    """
    Returns a table with winner_* and loser_* columns for every feature,
    plus the per-player history rows (used for the app's "current form").
    """
    with_elo, current_elo = add_elo_ratings(matches)
    serve, current_serve = add_serve_stats(matches)

    rows = add_history_features(player_rows(matches))
    hist = rows.set_index(["match_id", "won"])[HISTORY_FEATURES + ["career_matches"]]
    w_hist = hist.xs(1, level="won").add_prefix("winner_")
    l_hist = hist.xs(0, level="won").add_prefix("loser_")

    table = (with_elo.set_index("match_id")
             .join(serve.set_index("match_id"))
             .join(w_hist).join(l_hist)
             .reset_index())
    table = pd.concat([table, add_head_to_head(table)], axis=1)

    # Blended Elo (average of overall + surface), the best version from Phase 2.
    for side in ("winner", "loser"):
        table[f"{side}_blend_elo"] = blended(table[f"{side}_elo"], table[f"{side}_surface_elo"])
        table[f"{side}_surface_elo"] = table[f"{side}_surface_elo"].fillna(table[f"{side}_elo"])
        # Ranking: log scale, because the gap between #1 and #10 matters far
        # more than between #500 and #510. Unranked players get 2000.
        table[f"{side}_log_rank"] = np.log(table[f"{side}_rank"].fillna(2000))

    return table, rows, current_elo, current_serve


# Each "side" feature that becomes an A-minus-B difference.
SIDE_FEATURES = (["elo", "surface_elo", "blend_elo", "log_rank", "age", "h2h_wins"]
                 + HISTORY_FEATURES + STAT_NAMES)


# The exact columns the models use: all the differences plus a few facts about
# the match itself (a best-of-5 match makes the stronger player safer, etc.).
MODEL_FEATURES = ([f"diff_{f}" for f in SIDE_FEATURES]
                  + ["best_of", "min_career_matches", "min_stats_matches", "h2h_matches"])

# Plain-English names for charts and the app, and a group for each feature.
FEATURE_INFO = {
    "diff_elo":               ("Overall Elo rating", "Ratings & ranking"),
    "diff_surface_elo":       ("Surface Elo rating", "Ratings & ranking"),
    "diff_blend_elo":         ("Blended Elo rating", "Ratings & ranking"),
    "diff_log_rank":          ("ATP ranking", "Ratings & ranking"),
    "diff_age":               ("Age", "Match context"),
    "diff_h2h_wins":          ("Head-to-head wins", "Head-to-head"),
    "diff_days_since_last":   ("Days since last match", "Fatigue, rust & injury"),
    "diff_minutes_14d":       ("Minutes played (last 14 days)", "Fatigue, rust & injury"),
    "diff_matches_14d":       ("Matches played (last 14 days)", "Fatigue, rust & injury"),
    "diff_retirements_90d":   ("Retirements (last 90 days)", "Fatigue, rust & injury"),
    "diff_wins_last10":       ("Wins in last 10 matches", "Recent form"),
    "diff_first_serve_in":    ("1st serve in %", "Serve & return"),
    "diff_first_serve_won":   ("1st serve points won %", "Serve & return"),
    "diff_second_serve_in":   ("2nd serve in %", "Serve & return"),
    "diff_second_serve_won":  ("2nd serve points won %", "Serve & return"),
    "diff_serve_points_won":  ("Serve points won %", "Serve & return"),
    "diff_ace_rate":          ("Ace rate", "Serve & return"),
    "diff_double_fault_rate": ("Double fault rate", "Serve & return"),
    "diff_bp_saved":          ("Break points saved %", "Serve & return"),
    "diff_return_points_won": ("Return points won %", "Serve & return"),
    "diff_bp_converted":      ("Break points converted %", "Serve & return"),
    "diff_total_points_won":  ("Total points won %", "Serve & return"),
    "best_of":                ("Best of 3 or 5 sets", "Match context"),
    "min_career_matches":     ("Experience of the less experienced player", "Match context"),
    "min_stats_matches":      ("Stats history available", "Match context"),
    "h2h_matches":            ("Previous meetings", "Head-to-head"),
}
FEATURE_LABELS = {k: v[0] for k, v in FEATURE_INFO.items()}
FEATURE_GROUPS = {k: v[1] for k, v in FEATURE_INFO.items()}


def assign_players(table: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Randomly make the winner either Player A or Player B, then build differences."""
    rng = np.random.default_rng(seed)
    a_is_winner = rng.random(len(table)) < 0.5

    def pick(col_a_if_winner, col_a_if_loser):
        return np.where(a_is_winner, table[col_a_if_winner], table[col_a_if_loser])

    out = pd.DataFrame({
        "match_id": table["match_id"],
        "date": table["tourney_date"],
        "tourney_name": table["tourney_name"],
        "tourney_level": table["tourney_level"],
        "round": table["round"],
        "surface": table["surface"],
        "best_of": table["best_of"],
        "is_tour_match": table["is_tour_match"],
        "player_a": pick("winner_name", "loser_name"),
        "player_b": pick("loser_name", "winner_name"),
        "player_a_id": pick("winner_id", "loser_id"),
        "player_b_id": pick("loser_id", "winner_id"),
        "score": table["score"],
        "a_won": a_is_winner.astype(int),   # <- the answer the model tries to predict
    })

    for f in SIDE_FEATURES:
        a = pick(f"winner_{f}", f"loser_{f}")
        b = pick(f"loser_{f}", f"winner_{f}")
        out[f"diff_{f}"] = a - b

    # A few features that describe the match itself.
    out["elo_prob_a"] = win_probability(pick("winner_blend_elo", "loser_blend_elo"),
                                        pick("loser_blend_elo", "winner_blend_elo"))
    out["h2h_matches"] = table["winner_h2h_wins"] + table["loser_h2h_wins"]
    out["min_stats_matches"] = np.minimum(table["winner_stats_matches"], table["loser_stats_matches"])
    out["min_career_matches"] = np.minimum(table["winner_career_matches"], table["loser_career_matches"])
    return out


def current_form(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Each player's history features as they stand AFTER their latest match,
    for use in the app. (Fatigue is measured relative to the last date in
    the data, so players who haven't played recently show zero fatigue.)
    """
    last_date = rows["date"].max()
    g = rows.groupby("player_id")
    recent = lambda days: rows[rows["date"] > last_date - pd.Timedelta(days=days)].groupby("player_id")
    out = pd.DataFrame({
        "last_match": g["date"].max(),
        "days_since_last": (last_date - g["date"].max()).dt.days.clip(upper=MAX_REST_DAYS),
        "minutes_14d": recent(14)["minutes"].sum(),
        "matches_14d": recent(14).size(),
        "retirements_90d": recent(90)["retired"].sum(),
        "wins_last10": g["won"].apply(lambda s: s.tail(10).sum()),
        "career_matches": g.size(),
    })
    return out.fillna({"minutes_14d": 0, "matches_14d": 0, "retirements_90d": 0})


def player_snapshot(matches: pd.DataFrame, rows: pd.DataFrame,
                    current_elo: pd.DataFrame, current_serve: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player with everything the app needs to describe them "today"
    (= the last date in the data): ratings, serve stats, form, rank and age.
    """
    data_end = matches["tourney_date"].max()

    # Latest rank / age / country / hand, from each player's most recent match.
    sides = []
    for side in ("winner", "loser"):
        part = matches[["tourney_date", "is_tour_match", f"{side}_id", f"{side}_rank",
                        f"{side}_age", f"{side}_ioc", f"{side}_hand"]].copy()
        part.columns = ["date", "is_tour_match", "player_id", "rank", "age", "country", "hand"]
        sides.append(part)
    appearances = pd.concat(sides).sort_values("date", kind="mergesort")
    latest = appearances.groupby("player_id").last()   # last non-missing value of each column
    latest["age"] = latest["age"] + (data_end - appearances.groupby("player_id")["date"].max()).dt.days / 365.25
    latest["last_tour_match"] = appearances[appearances["is_tour_match"]].groupby("player_id")["date"].max()

    snapshot = (current_elo.set_index("player_id")
                .join(current_serve)
                .join(current_form(rows).drop(columns="last_match"))
                .join(latest[["rank", "age", "country", "hand", "last_tour_match"]]))
    snapshot["log_rank"] = np.log(snapshot["rank"].fillna(2000))
    return snapshot.reset_index()


def head_to_head_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Total wins for every pair of players who have met (lower id listed first)."""
    lo = np.minimum(matches["winner_id"], matches["loser_id"])
    hi = np.maximum(matches["winner_id"], matches["loser_id"])
    lo_won = (matches["winner_id"] == lo).astype(int)
    h2h = pd.DataFrame({"lo_id": lo, "hi_id": hi, "lo_wins": lo_won, "hi_wins": 1 - lo_won})
    return h2h.groupby(["lo_id", "hi_id"], as_index=False).sum()


def build_model_table(save: bool = True) -> pd.DataFrame:
    """Run Phases 1-3 end to end and return the model table."""
    matches = load_matches(save=False)
    table, rows, current_elo, current_serve = build_player_side_table(matches)
    model_table = assign_players(table)
    if save:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        model_table.to_csv(MODEL_TABLE_FILE, index=False)
        snapshot = player_snapshot(matches, rows, current_elo, current_serve)
        numeric = snapshot.select_dtypes("number").columns
        snapshot[numeric] = snapshot[numeric].round(5)   # shorter file
        snapshot.to_csv(PLAYERS_FILE, index=False, date_format="%Y-%m-%d")
        head_to_head_table(matches).to_csv(H2H_FILE, index=False)
    return model_table


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def summarize(mt: pd.DataFrame) -> None:
    print(f"\n=== Model table: {len(mt):,} matches, {mt.filter(like='diff_').shape[1]} difference features ===")
    print(f"Player A won {mt['a_won'].mean():.1%} of matches (should be close to 50%)")

    # Look at the TRAINING years only (2000-2021, main tour), so we don't
    # peek at the test years before Phase 4.
    train = mt[(mt["date"] < "2022-01-01") & mt["is_tour_match"]]
    print(f"\nTraining years, main tour ({len(train):,} matches):")
    print("How often Player A won when A had the HIGHER value of each feature")
    print("(above 50% = higher helps, below 50% = higher hurts):")
    rows = []
    for col in [c for c in mt.columns if c.startswith("diff_")]:
        d = train[col]
        mask = d.notna() & (d != 0)
        rows.append((col.removeprefix("diff_"), (train.loc[mask, "a_won"] == (d[mask] > 0)).mean(), mask.mean()))
    for name, rate, share in sorted(rows, key=lambda r: -abs(r[1] - 0.5)):
        print(f"  {name:<20} {rate:6.1%}   (players differ in {share:.0%} of matches)")


if __name__ == "__main__":
    summarize(build_model_table())
    print(f"\nSaved: {MODEL_TABLE_FILE.name}, {PLAYERS_FILE.name}, {H2H_FILE.name}")
