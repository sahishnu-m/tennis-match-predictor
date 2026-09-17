"""
Predict a match from the command line.

Examples:
    python -m src.predict "Alcaraz" "Sinner" --surface Hard --best-of 5
    python -m src.predict "Djokovic" "Zverev" --surface Clay

    # Replay a past match from the test years (2023+):
    python -m src.predict --replay "US Open" --year 2025 --round F

Needs the files in app_data/ (created by `python -m src.train`).
"""

import argparse

import pandas as pd

from src.data_loader import APP_DATA_DIR
from src.matchup import SURFACES, Predictor


def show_prediction(pred: Predictor, name_a: str, name_b: str, surface: str, best_of: int):
    a, b = pred.find(name_a), pred.find(name_b)
    pa, pb = pred.players.loc[a], pred.players.loc[b]
    result = pred.predict(a, b, surface, best_of)
    p = result["prob_a"]

    print(f"\n{pa['name']} vs {pb['name']} | {surface} | best of {best_of} "
          f"| player data as of {pred.data_end:%Y-%m-%d}")
    print(f"\n  {pa['name']:<25} {p:6.1%}  {'#' * round(p * 40)}")
    print(f"  {pb['name']:<25} {1 - p:6.1%}  {'#' * round((1 - p) * 40)}")
    print(f"\n  Other models: logistic regression {result['prob_logreg']:.1%}, "
          f"Elo only {result['prob_elo']:.1%} (for {pa['name']})")

    print("\n  Biggest factors:")
    for f in result["factors"]:
        who = pa["name"] if f.impact > 0 else pb["name"]
        values = f"  ({f.value_a} vs {f.value_b})" if f.value_a else ""
        print(f"    {f.label:<35} favours {who}{values}")


def replay(pred: Predictor, tournament: str, year: int | None, round_: str | None):
    tests = pd.read_csv(APP_DATA_DIR / "test_predictions.csv", parse_dates=["date"])
    found = tests[tests["tourney_name"].str.contains(tournament, case=False, regex=False)]
    if year:
        found = found[found["date"].dt.year == year]
    if round_:
        found = found[found["round"] == round_.upper()]
    if found.empty:
        raise SystemExit("No matching test-set match (the replay covers main-tour matches from 2023 on).")

    for _, m in found.head(10).iterrows():
        winner, loser = (m["player_a"], m["player_b"]) if m["a_won"] else (m["player_b"], m["player_a"])
        p_winner = m["prob_xgb"] if m["a_won"] else 1 - m["prob_xgb"]
        verdict = "model got it RIGHT" if p_winner > 0.5 else "UPSET (model picked the other player)"
        print(f"\n{m['date']:%Y} {m['tourney_name']} {m['round']}: {winner} def. {loser} {m['score']}")
        print(f"  Before the match, the model gave {winner} a {p_winner:.1%} chance -> {verdict}")
    if len(found) > 10:
        print(f"\n({len(found) - 10} more matches; narrow it down with --year / --round)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a tennis match.")
    parser.add_argument("players", nargs="*", help='two names, e.g. "Alcaraz" "Sinner"')
    parser.add_argument("--surface", default="Hard", choices=SURFACES)
    parser.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    parser.add_argument("--replay", metavar="TOURNAMENT", help="replay a past test-set match")
    parser.add_argument("--year", type=int)
    parser.add_argument("--round", dest="round_", help="e.g. F, SF, QF, R16")
    args = parser.parse_args()

    predictor = Predictor()
    if args.replay:
        replay(predictor, args.replay, args.year, args.round_)
    elif len(args.players) == 2:
        show_prediction(predictor, *args.players, args.surface, args.best_of)
    else:
        parser.error('give two player names, or use --replay "Tournament"')
