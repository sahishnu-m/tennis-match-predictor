"""
Serve and return statistics for every player, before every match.

The raw data has per-match counts such as:
    w_svpt    serve points the winner played
    w_1stIn   first serves that went in
    w_1stWon  first-serve points won
    w_2ndWon  second-serve points won
    w_df      double faults
    w_ace     aces
    w_bpSaved / w_bpFaced   break points saved / faced
(and the same with l_ for the loser).

THE LEAKAGE TRAP
These numbers are only known AFTER a match ends. If we used a match's own
serve stats to predict that match, the model would be cheating (the winner
almost always wins more serve points). So for each match we use each player's
stats from their previous 20 matches ONLY.

Run it:
    python -m src.serve_stats
"""

import numpy as np
import pandas as pd

from src.data_loader import PROCESSED_DIR, load_matches

WINDOW = 20          # look back over a player's last 20 matches that have stats
PRIOR_MATCHES = 2    # how strongly to pull small samples toward the tour average

SERVE_STATS_FILE = PROCESSED_DIR / "serve_stats.csv"
CURRENT_SERVE_FILE = PROCESSED_DIR / "current_serve_stats.csv"

# The raw count columns, from ONE player's point of view.
# "o_" means the opponent's numbers in the same match.
COUNT_COLUMNS = ["svpt", "first_in", "first_won", "second_won", "df", "ace",
                 "bp_saved", "bp_faced",
                 "o_svpt", "o_first_won", "o_second_won", "o_bp_saved", "o_bp_faced"]

# Map the raw file's column names (without the w_/l_ prefix) to ours.
RAW_NAMES = {"svpt": "svpt", "1stIn": "first_in", "1stWon": "first_won",
             "2ndWon": "second_won", "df": "df", "ace": "ace",
             "bpSaved": "bp_saved", "bpFaced": "bp_faced"}

def stat_parts(d: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """
    Every statistic is a ratio. Return {stat name: (numerator, denominator)}
    for a table of counts. Keeping top and bottom separate lets us add up
    several matches properly before dividing.
    """
    second_serves = d.svpt - d.first_in                    # points that needed a 2nd serve
    serve_won = d.first_won + d.second_won                 # serve points won
    return_won = d.o_svpt - d.o_first_won - d.o_second_won  # opponent's serve points we won
    return {
        # --- serving ---
        "first_serve_in":    (d.first_in, d.svpt),                   # 1st serve %
        "first_serve_won":   (d.first_won, d.first_in),              # points won when 1st serve goes in
        "second_serve_in":   (second_serves - d.df, second_serves),  # 2nd serves that land in
        "second_serve_won":  (d.second_won, second_serves),          # points won on 2nd serve
        "serve_points_won":  (serve_won, d.svpt),                    # all serve points won
        "ace_rate":          (d.ace, d.svpt),                        # aces per serve point
        "double_fault_rate": (d.df, d.svpt),                         # double faults per serve point
        "bp_saved":          (d.bp_saved, d.bp_faced),               # break points saved
        # --- returning ---
        "return_points_won": (return_won, d.o_svpt),                 # points won on opponent's serve
        "bp_converted":      (d.o_bp_faced - d.o_bp_saved, d.o_bp_faced),  # break chances taken
        # --- overall ---
        "total_points_won":  (serve_won + return_won, d.svpt + d.o_svpt),
    }


STAT_NAMES = list(stat_parts(pd.DataFrame(columns=COUNT_COLUMNS)).keys())


# ---------------------------------------------------------------------------
# Step 1: one row per player per match ("long" format)
# ---------------------------------------------------------------------------

def to_player_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Each match becomes two rows: one from the winner's view, one from the loser's.
    That way "a player's history" is simply all rows with their player_id.
    """
    halves = []
    for me, opp, won in (("w", "l", True), ("l", "w", False)):
        half = pd.DataFrame({
            "match_id": matches["match_id"].to_numpy(),
            "player_id": matches[f"{'winner' if won else 'loser'}_id"].to_numpy(),
            "won": won,
        })
        for raw, ours in RAW_NAMES.items():
            half[ours] = matches[f"{me}_{raw}"].to_numpy()
        for raw in ("svpt", "1stWon", "2ndWon", "bpSaved", "bpFaced"):
            half[f"o_{RAW_NAMES[raw]}"] = matches[f"{opp}_{raw}"].to_numpy()
        halves.append(half)

    rows = pd.concat(halves, ignore_index=True)

    # Some rows have missing or impossible numbers (e.g. more first serves in than
    # serves hit). Mark those as "no stats" so they can't confuse the averages.
    d = rows
    valid = (
        d[COUNT_COLUMNS].notna().all(axis=1)
        & (d.svpt > 0) & (d.o_svpt > 0)
        & (d.first_in <= d.svpt) & (d.first_won <= d.first_in)
        & (d.second_won + d.df <= d.svpt - d.first_in)
        & (d.o_first_won + d.o_second_won <= d.o_svpt)
        & (d.bp_saved <= d.bp_faced) & (d.o_bp_saved <= d.o_bp_faced)
    )
    rows["has_stats"] = valid
    # Rows without stats contribute zero to every total.
    rows.loc[~valid, COUNT_COLUMNS] = 0
    rows[COUNT_COLUMNS] = rows[COUNT_COLUMNS].astype(float)
    return rows.sort_values(["player_id", "match_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: rolling totals, using ONLY earlier matches
# ---------------------------------------------------------------------------

def rolling_player_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """
    For every row, compute the player's stats over their previous WINDOW matches
    that had stats, NOT counting the current match.
    """
    parts = stat_parts(rows)
    num_cols, den_cols = [], []
    for name, (num, den) in parts.items():
        rows[f"{name}__num"] = num
        rows[f"{name}__den"] = den
        num_cols.append(f"{name}__num")
        den_cols.append(f"{name}__den")
    value_cols = num_cols + den_cols + ["has_stats"]

    # Keep only rows that have stats, then take rolling sums over the last WINDOW
    # of them. Trick: rolling sum = running total now - running total WINDOW rows ago.
    s = rows.loc[rows["has_stats"], ["player_id"] + value_cols].copy()
    s["has_stats"] = s["has_stats"].astype(float)
    running = s.groupby("player_id")[value_cols].cumsum()
    earlier = running.groupby(s["player_id"]).shift(WINDOW).fillna(0)
    after_match = running - earlier   # totals INCLUDING this match

    # Put those "after this match" totals back on the full table, then shift by one
    # row per player so each match sees only the totals from before it.
    state = pd.DataFrame(np.nan, index=rows.index, columns=value_cols)
    state.loc[after_match.index] = after_match
    before = state.groupby(rows["player_id"]).shift(1)
    before = before.groupby(rows["player_id"]).ffill()   # carry forward past no-stats matches
    before = before.fillna(0)                              # brand-new players: nothing yet

    # Tour-wide averages, again only from earlier matches. These are the "prior"
    # we pull small samples toward, so a player with 1 match doesn't get a crazy %.
    by_match = rows.groupby("match_id")[value_cols].sum()
    cum = by_match.cumsum().shift(1).fillna(0)       # totals before each match
    prior = cum.loc[rows["match_id"]].reset_index(drop=True)
    seen_rows = prior["has_stats"].replace(0, np.nan)

    out = rows[["match_id", "player_id"]].copy()
    out["stats_matches"] = before["has_stats"]
    for name in parts:
        num, den = before[f"{name}__num"], before[f"{name}__den"]
        prior_rate = prior[f"{name}__num"] / prior[f"{name}__den"].replace(0, np.nan)
        prior_weight = PRIOR_MATCHES * prior[f"{name}__den"] / seen_rows  # ~2 matches' worth
        # Weighted blend: mostly the player's own numbers once they have a few matches.
        out[name] = (num + prior_weight * prior_rate) / (den + prior_weight)
    return out


# ---------------------------------------------------------------------------
# Step 3: attach to matches as winner_... and loser_... columns
# ---------------------------------------------------------------------------

def add_serve_stats(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = to_player_rows(matches)
    stats = rolling_player_stats(rows)
    stats["won"] = rows["won"].to_numpy()

    cols = STAT_NAMES + ["stats_matches"]
    winners = stats[stats["won"]].set_index("match_id")[cols].add_prefix("winner_")
    losers = stats[~stats["won"]].set_index("match_id")[cols].add_prefix("loser_")
    per_match = winners.join(losers).reset_index().sort_values("match_id")

    # Each player's stats including their most recent match (for the app).
    return per_match, latest_stats(rows)


def latest_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Each player's stats over their last WINDOW matches with stats (including the newest)."""
    s = rows[rows["has_stats"]].groupby("player_id").tail(WINDOW)
    parts = stat_parts(s)
    totals = pd.DataFrame({f"{k}__num": v[0] for k, v in parts.items()} |
                          {f"{k}__den": v[1] for k, v in parts.items()})
    totals["player_id"] = s["player_id"]
    g = totals.groupby("player_id").sum()
    out = pd.DataFrame({k: g[f"{k}__num"] / g[f"{k}__den"].replace(0, np.nan) for k in parts})
    out["stats_matches"] = s.groupby("player_id").size()
    return out


# ---------------------------------------------------------------------------
# Checkpoint: does each stat, on its own, pick the winner?
# ---------------------------------------------------------------------------

def evaluate(matches: pd.DataFrame, per_match: pd.DataFrame, since: str = "2023-01-01") -> None:
    m = matches[["match_id", "tourney_date", "is_tour_match"]].merge(per_match, on="match_id")
    test = m[(m["tourney_date"] >= since) & m["is_tour_match"]]
    test = test[(test["winner_stats_matches"] >= 5) & (test["loser_stats_matches"] >= 5)]
    print(f"\n=== Checkpoint: main-tour matches since {since}, both players with 5+ matches of stats "
          f"({len(test):,}) ===")
    print("How often the player with the BETTER past stat won:")
    lower_is_better = {"double_fault_rate"}
    results = []
    for name in STAT_NAMES:
        w, l = test[f"winner_{name}"], test[f"loser_{name}"]
        ok = w.notna() & l.notna() & (w != l)
        better = (w[ok] < l[ok]) if name in lower_is_better else (w[ok] > l[ok])
        results.append((name, better.mean()))
    for name, acc in sorted(results, key=lambda r: -r[1]):
        print(f"  {name:<20} {acc:.1%}")


if __name__ == "__main__":
    matches = load_matches(save=False)
    per_match, current = add_serve_stats(matches)
    per_match.to_csv(SERVE_STATS_FILE, index=False)
    current.to_csv(CURRENT_SERVE_FILE)
    evaluate(matches, per_match)

    names = pd.concat([matches[["winner_id", "winner_name"]].set_axis(["player_id", "name"], axis=1),
                       matches[["loser_id", "loser_name"]].set_axis(["player_id", "name"], axis=1)]
                      ).drop_duplicates("player_id", keep="last").set_index("player_id")["name"]
    show = current.join(names).set_index("name")
    top = ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Alexander Zverev"]
    print("\nCurrent stats (last 20 matches) for a few players:")
    print((show.loc[show.index.intersection(top)] * ([100] * len(STAT_NAMES) + [1])).round(1)
          .T.rename(index={"stats_matches": "matches used"}).to_string())
    print(f"\nSaved: {SERVE_STATS_FILE.name}, {CURRENT_SERVE_FILE.name}")
