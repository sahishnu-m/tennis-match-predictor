"""
Phase 1: download and clean the ATP match data.

Data source: Jeff Sackmann's tennis_atp repository
    https://github.com/JeffSackmann/tennis_atp
Licensed CC BY-NC-SA 4.0 (free to use for non-commercial projects, with credit).

We load two kinds of files for every year from 2000 onward:
  - atp_matches_YYYY.csv             main tour (Grand Slams, Masters, ATP 250/500, Davis Cup...)
  - atp_matches_qual_chall_YYYY.csv  tour-level qualifying + Challenger events

The second kind adds many more players and gives young players a history
before they reach the main tour. Predictions are still *tested* only on
main-tour matches (the "is_tour_match" column), so results stay comparable.

Run this file directly to download the data and print some basic stats:
    python -m src.data_loader
"""

from datetime import date
from pathlib import Path
import urllib.error
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Folder layout: data/raw holds the untouched CSVs, data/processed holds our cleaned file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_FILE = PROCESSED_DIR / "matches_clean.csv"

# Small files the web app needs. Unlike data/, this folder IS uploaded to
# GitHub, so the deployed app works without re-running the whole pipeline.
APP_DATA_DIR = PROJECT_ROOT / "app_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The files live in one of these repos. We try the original first. As of
# September 2026 it is no longer public on GitHub, so we fall back to public
# forks. Their newest commit is Jeff Sackmann's own (2026-06-08), so they hold
# exactly his data.
SOURCE_REPOS = [
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/",
    "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/",
    "https://raw.githubusercontent.com/racketbracket/tennis_atp/master/",
]

# The two kinds of file we download: {label: file name pattern}.
FILE_KINDS = {
    "tour": "atp_matches_{year}.csv",
    "qual_chall": "atp_matches_qual_chall_{year}.csv",
}

FIRST_YEAR = 2000

# Inside a tournament, every match has the same "tourney_date" (the start date
# of the event). To put matches in the order they were really played, we sort
# by round: qualifying first, then early rounds, the final last.
ROUND_ORDER = {
    "Q1": -4,  # qualifying rounds (played before the main draw)
    "Q2": -3,
    "Q3": -2,
    "Q4": -1,
    "ER": 0,   # "early rounds" (a handful of events use this label)
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "RR": 4,   # round robin (e.g. ATP Finals group stage)
    "QF": 5,
    "SF": 6,
    "BR": 7,   # bronze medal match (Olympics)
    "F": 8,
}


# ---------------------------------------------------------------------------
# Step 1: download
# ---------------------------------------------------------------------------

def download_file(filename: str, path: Path) -> bool:
    """
    Try each source repo until one has this file.
    Returns True if it was downloaded, False if no source has it (404 everywhere).
    """
    for repo in SOURCE_REPOS:
        try:
            urllib.request.urlretrieve(repo + filename, path)
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue  # missing at this address: try the next one
            raise  # some other problem: stop and show the error
    return False


def download_data(force: bool = False) -> dict[str, list[Path]]:
    """
    Download every file kind for every year from 2000 onward.

    The last year is found automatically: we keep trying the next year
    until no source has the file. Files that are already on disk are skipped
    unless force=True.

    Returns {kind: [paths]}, e.g. {"tour": [...], "qual_chall": [...]}.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {}

    for kind, pattern in FILE_KINDS.items():
        files[kind] = []
        for year in range(FIRST_YEAR, date.today().year + 1):
            filename = pattern.format(year=year)
            path = RAW_DIR / filename

            if path.exists() and not force:
                files[kind].append(path)
                continue

            print(f"Downloading {filename}...")
            if download_file(filename, path):
                files[kind].append(path)
            else:
                # No file for this year yet, so we've reached the end.
                print(f"No {kind} file for {year} yet. Latest available year is {year - 1}.")
                break

    return files


# ---------------------------------------------------------------------------
# Step 2: load + clean
# ---------------------------------------------------------------------------

def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw combined CSVs into one tidy, chronologically sorted table."""
    df = raw.copy()

    # Dates are stored as numbers like 20230116. Convert them to real dates.
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")

    # Scores are text like "6-4 3-6 7-6(5)". Make sure missing ones are empty strings
    # so the text checks below don't crash.
    df["score"] = df["score"].fillna("").astype(str).str.strip()

    # Walkovers ("W/O") mean the match was never played, so they tell us nothing
    # about who is better. Remove them. We also remove matches with no score at all.
    is_walkover = df["score"].str.contains("W/O", case=False)
    is_blank = df["score"] == ""
    df = df[~is_walkover & ~is_blank].copy()

    # Retirements ("RET") mean a player quit mid-match, usually because of injury.
    # We keep these matches but flag them; they become an injury signal in Phase 3.
    df["is_retirement"] = df["score"].str.contains("RET", case=False)

    # Surfaces: keep the name tidy (Hard, Clay, Grass, Carpet) and mark blanks.
    df["surface"] = df["surface"].fillna("Unknown").str.strip().str.title()

    # True for main-tour matches. Models are evaluated on these only.
    df["is_tour_match"] = df["source"] == "tour"

    # Put matches in the order they were actually played:
    # tournament start date -> tournament -> round -> match number.
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(4)
    df = df.sort_values(
        ["tourney_date", "tourney_id", "round_order", "match_num"],
        kind="mergesort",  # a "stable" sort: ties keep their original order
    ).reset_index(drop=True)

    # A simple running number (0, 1, 2, ...) that means "the n-th match in history".
    # Later phases use it to make sure features only look at earlier matches.
    df["match_id"] = range(len(df))

    return df


def load_matches(force_download: bool = False, save: bool = True) -> pd.DataFrame:
    """Download (if needed), combine all years, clean, and return one DataFrame."""
    files = download_data(force=force_download)
    if not files["tour"]:
        raise RuntimeError("No data files found. Check your internet connection.")

    # Read every file, remember which kind it was, and stack them into one table.
    tables = []
    for kind, paths in files.items():
        for f in paths:
            table = pd.read_csv(f, low_memory=False)
            table["source"] = kind
            tables.append(table)
    raw = pd.concat(tables, ignore_index=True)
    raw_count = len(raw)

    df = clean_matches(raw)

    # Save a copy so you can open it in VS Code and look around.
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_FILE, index=False)

    df.attrs["rows_removed"] = raw_count - len(df)  # remember how many we dropped
    return df


# ---------------------------------------------------------------------------
# Step 3: print a quick summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    tour = df[df["is_tour_match"]]
    print("\n=== ATP match data summary ===")
    print(f"Matches:            {len(df):,}  (main tour {len(tour):,}, qualifying/Challenger {len(df) - len(tour):,})")
    print(f"Removed (walkovers / no score): {df.attrs.get('rows_removed', 0):,}")
    print(f"Date range:         {df['tourney_date'].min():%Y-%m-%d} to {df['tourney_date'].max():%Y-%m-%d}")
    print(f"Unique players:     {pd.concat([df['winner_id'], df['loser_id']]).nunique():,}  "
          f"(main tour only: {pd.concat([tour['winner_id'], tour['loser_id']]).nunique():,})")
    print(f"Retirements:        {df['is_retirement'].sum():,} ({df['is_retirement'].mean():.1%})")

    print("\nMatches by surface:")
    print(df["surface"].value_counts().to_string())

    # How many matches have serve stats (aces, first serves, etc.)?
    has_stats = df["w_svpt"].notna()
    print("\nMatches per year (and % with serve stats):")
    per_year = df.assign(year=df["tourney_date"].dt.year, has_stats=has_stats).groupby(["year", "source"]).agg(
        matches=("match_id", "size"), with_stats=("has_stats", "mean"))
    print(per_year.unstack("source").to_string(
        formatters={c: ("{:.0%}".format if c[0] == "with_stats" else "{:,.0f}".format) for c in per_year.unstack("source").columns}))

    print(f"\nClean data saved to: {CLEAN_FILE}")


if __name__ == "__main__":
    matches = load_matches()
    print_summary(matches)
