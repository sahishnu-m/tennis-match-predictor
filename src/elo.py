"""
Phase 2: Elo ratings.

Elo is a rating system first built for chess. The idea:
  - Everyone starts at 1500.
  - Before a match, the two ratings give each player an expected chance to win.
  - After the match, the winner takes points from the loser. Beating someone
    much stronger earns lots of points; beating someone weaker earns few.

We keep two kinds of rating for each player:
  - overall Elo   (every match counts)
  - surface Elo   (separate ratings for Hard, Clay and Grass)

Run it:
    python -m src.elo                                   # build ratings + checkpoint
    python -m src.elo --predict "Alcaraz" "Sinner" --surface Hard
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from src.data_loader import PROCESSED_DIR, load_matches

STARTING_ELO = 1500
ELO_SURFACES = ["Hard", "Clay", "Grass"]  # Carpet/unknown only affect overall Elo

MATCHES_WITH_ELO_FILE = PROCESSED_DIR / "matches_elo.csv"
CURRENT_ELO_FILE = PROCESSED_DIR / "current_elo.csv"


# ---------------------------------------------------------------------------
# The two formulas at the heart of Elo
# ---------------------------------------------------------------------------

def win_probability(elo_a, elo_b):
    """
    Chance that player A beats player B, based only on their ratings.
    A 200-point gap is about 76%; a 400-point gap is about 91%.
    Works on single numbers or whole columns at once.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def k_factor(matches_played: int) -> float:
    """
    How much one result can move a rating.
    New players have a big K (their rating moves fast while we learn how good
    they are). Veterans have a small K (one result barely moves their rating).
        0 matches   -> K is about 132
        100 matches -> K is about 39
        500 matches -> K is about 21
    """
    return 250 / (matches_played + 5) ** 0.4


# ---------------------------------------------------------------------------
# Walk through history, one match at a time
# ---------------------------------------------------------------------------

def add_elo_ratings(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Go through every match in date order. For each match we:
      1. WRITE DOWN both players' ratings *before* the match (no peeking).
      2. THEN update the ratings using the result.

    Returns:
      - the matches table with new pre-match rating columns
      - a table of every player's latest ("current") ratings
    """
    # Ratings and match counts, looked up by player id.
    # defaultdict gives new players the starting value automatically.
    elo = defaultdict(lambda: STARTING_ELO)
    played = defaultdict(int)
    surf_elo = {s: defaultdict(lambda: STARTING_ELO) for s in ELO_SURFACES}
    surf_played = {s: defaultdict(int) for s in ELO_SURFACES}

    # Lists that will become new columns (one value per match).
    w_elo, l_elo, w_surf, l_surf = [], [], [], []
    w_count, l_count = [], []

    # Also remember each player's name and last match date for the app later.
    last_name, last_date = {}, {}

    for m in matches.itertuples(index=False):
        w, l = m.winner_id, m.loser_id

        # --- 1. Record pre-match ratings -----------------------------------
        w_elo.append(elo[w])
        l_elo.append(elo[l])
        w_count.append(played[w])
        l_count.append(played[l])

        has_surface = m.surface in surf_elo
        if has_surface:
            se, sp = surf_elo[m.surface], surf_played[m.surface]
            w_surf.append(se[w])
            l_surf.append(se[l])
        else:
            w_surf.append(np.nan)  # no surface rating for Carpet / unknown
            l_surf.append(np.nan)

        # --- 2. Update ratings with the result -----------------------------
        # Expected chance the winner would win. If they were a big favourite
        # this is close to 1, so they gain only a little.
        expected = win_probability(elo[w], elo[l])
        elo[w] += k_factor(played[w]) * (1 - expected)
        elo[l] -= k_factor(played[l]) * (1 - expected)
        played[w] += 1
        played[l] += 1

        # Same update, but only using matches on this surface.
        if has_surface:
            expected = win_probability(se[w], se[l])
            se[w] += k_factor(sp[w]) * (1 - expected)
            se[l] -= k_factor(sp[l]) * (1 - expected)
            sp[w] += 1
            sp[l] += 1

        for pid, name in ((w, m.winner_name), (l, m.loser_name)):
            last_name[pid] = name
            last_date[pid] = m.tourney_date

    # Attach the new columns to a copy of the matches table.
    out = matches.copy()
    out["winner_elo"] = w_elo
    out["loser_elo"] = l_elo
    out["winner_surface_elo"] = w_surf
    out["loser_surface_elo"] = l_surf
    out["winner_matches_played"] = w_count
    out["loser_matches_played"] = l_count

    # Build the "current ratings" table (ratings after the last match in the data).
    rows = []
    for pid in played:
        row = {
            "player_id": pid,
            "name": last_name[pid],
            "elo": elo[pid],
            "matches": played[pid],
            "last_match": last_date[pid],
        }
        for s in ELO_SURFACES:
            row[f"elo_{s.lower()}"] = surf_elo[s][pid]
            row[f"matches_{s.lower()}"] = surf_played[s][pid]
        rows.append(row)
    current = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)

    return out, current


# ---------------------------------------------------------------------------
# Checkpoint: how good is "the higher-rated player wins"?
# ---------------------------------------------------------------------------

def blended(overall, surface):
    """Average of overall and surface Elo (falls back to overall if no surface rating)."""
    return (overall + surface.fillna(overall)) / 2


def evaluate(df: pd.DataFrame, since: str = "2023-01-01") -> None:
    # Only test on main-tour matches. Qualifying/Challenger matches still
    # help build the ratings, but we judge the ratings on the main tour.
    test = df[(df["tourney_date"] >= since) & df["is_tour_match"]]
    print(f"\n=== Checkpoint: main-tour matches since {since} ({len(test):,} matches) ===")

    for label, wr, lr in [
        ("Overall Elo", test["winner_elo"], test["loser_elo"]),
        ("Surface Elo", test["winner_surface_elo"], test["loser_surface_elo"]),
        ("Blend (avg of both)",
         blended(test["winner_elo"], test["winner_surface_elo"]),
         blended(test["loser_elo"], test["loser_surface_elo"])),
    ]:
        valid = wr.notna() & lr.notna() & (wr != lr)  # skip exact ties
        wr, lr = wr[valid], lr[valid]

        # Accuracy: how often the higher-rated player was the winner.
        accuracy = (wr > lr).mean()

        # Log loss: punishes confident wrong guesses. Lower is better;
        # a coin flip (always 50%) scores 0.693.
        p_winner = win_probability(wr, lr).clip(1e-6, 1 - 1e-6)
        log_loss = -np.log(p_winner).mean()

        print(f"{label:<22} higher rating won {accuracy:.1%}   "
              f"log loss {log_loss:.3f}   (n={valid.sum():,})")


# ---------------------------------------------------------------------------
# Quick predictions using the latest ratings
# ---------------------------------------------------------------------------

def find_player(current: pd.DataFrame, query: str) -> pd.Series:
    """Find a player by (part of) their name, ignoring upper/lower case."""
    names = current["name"].str.lower()
    exact = current[names == query.lower()]
    if len(exact):
        return exact.iloc[0]
    partial = current[names.str.contains(query.lower(), regex=False)]
    if partial.empty:
        raise SystemExit(f"No player matches '{query}'.")
    if len(partial) > 1:
        # The table is sorted by Elo, so we pick the highest-rated match,
        # but tell the user about the others.
        others = ", ".join(partial["name"].head(5))
        print(f"'{query}' matches several players ({others}); using {partial.iloc[0]['name']}.")
    return partial.iloc[0]


def predict(current: pd.DataFrame, name_a: str, name_b: str, surface: str) -> None:
    a, b = find_player(current, name_a), find_player(current, name_b)
    col = f"elo_{surface.lower()}"
    # Blend overall and surface Elo, same as in the checkpoint.
    ra = (a["elo"] + a[col]) / 2
    rb = (b["elo"] + b[col]) / 2
    p = win_probability(ra, rb)

    as_of = pd.to_datetime(current["last_match"]).max()
    print(f"\n{a['name']} vs {b['name']} on {surface} (ratings as of {as_of:%Y-%m-%d})")
    for player, rating in ((a, ra), (b, rb)):
        print(f"  {player['name']:<25} overall {player['elo']:.0f} | "
              f"{surface} {player[col]:.0f} | blend {rating:.0f}")
    print(f"  Win chance: {a['name']} {p:.0%}  |  {b['name']} {1 - p:.0%}")


def print_top(current: pd.DataFrame, since: pd.Timestamp, n: int = 10) -> None:
    # Only show players who've played recently (retired players keep old ratings).
    active = current[current["last_match"] >= since]
    print(f"\nTop {n} active players by overall Elo:")
    print(active.head(n)[["name", "elo", "elo_hard", "elo_clay", "elo_grass", "matches"]]
          .round(0).to_string(index=False))


def build_elo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the matches, compute all ratings, and save both tables."""
    matches = load_matches()
    with_elo, current = add_elo_ratings(matches)
    with_elo.to_csv(MATCHES_WITH_ELO_FILE, index=False)
    current.to_csv(CURRENT_ELO_FILE, index=False)
    return with_elo, current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Elo ratings or predict a match.")
    parser.add_argument("--predict", nargs=2, metavar=("PLAYER_A", "PLAYER_B"))
    parser.add_argument("--surface", default="Hard", choices=ELO_SURFACES)
    args = parser.parse_args()

    with_elo, current = build_elo()

    if args.predict:
        predict(current, *args.predict, args.surface)
    else:
        evaluate(with_elo)
        print_top(current, since=with_elo["tourney_date"].max() - pd.Timedelta(days=365))
        print(f"\nSaved: {MATCHES_WITH_ELO_FILE.name}, {CURRENT_ELO_FILE.name}")
