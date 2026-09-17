"""
Predict a match between any two players, using their latest numbers.

This is shared by the command-line tool (src/predict.py) and the web app
(app.py). It only reads the small files in app_data/, so it works without
re-running the whole pipeline.

The main idea: build ONE row that looks exactly like a row of the training
table (Player A minus Player B), then hand it to the trained XGBoost model.
"""

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import APP_DATA_DIR
from src.elo import win_probability
from src.features import FEATURE_LABELS, HISTORY_FEATURES, MODEL_FEATURES
from src.serve_stats import STAT_NAMES

SURFACES = ["Hard", "Clay", "Grass"]


@dataclass
class Factor:
    """One reason behind a prediction."""
    feature: str
    label: str
    impact: float      # > 0 helps Player A, < 0 helps Player B (log-odds units)
    value_a: str
    value_b: str


def load_players() -> pd.DataFrame:
    players = pd.read_csv(APP_DATA_DIR / "players.csv",
                          parse_dates=["last_match", "last_tour_match"])
    # Players with no serve stats yet get the typical tour value, which is
    # what the training data gave them too (the "prior" in serve_stats.py).
    has_stats = players["stats_matches"] >= 10
    players[STAT_NAMES] = players[STAT_NAMES].fillna(players.loc[has_stats, STAT_NAMES].median())
    players["stats_matches"] = players["stats_matches"].fillna(0)
    return players.set_index("player_id")


class Predictor:
    def __init__(self):
        self.players = load_players()
        h2h = pd.read_csv(APP_DATA_DIR / "h2h.csv")
        self.h2h = {(r.lo_id, r.hi_id): (r.lo_wins, r.hi_wins) for r in h2h.itertuples()}
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(APP_DATA_DIR / "xgb_model.json")
        self.logreg = joblib.load(APP_DATA_DIR / "logreg.joblib")
        self.metrics = json.loads((APP_DATA_DIR / "metrics.json").read_text(encoding="utf-8"))
        self.data_end = pd.Timestamp(self.metrics["data_end"])

    # -- looking players up --------------------------------------------------

    def find(self, query: str) -> int:
        """Player id for a (partial) name. Prefers tour players with higher Elo."""
        p = self.players.sort_values("elo", ascending=False)
        names = p["name"].str.lower()
        exact = p[names == query.lower()]
        if len(exact):
            return exact.index[0]
        partial = p[names.str.contains(query.lower(), regex=False)]
        if partial.empty:
            raise ValueError(f"No player matches '{query}'.")
        with_tour = partial[partial["last_tour_match"].notna()]
        return (with_tour if len(with_tour) else partial).index[0]

    def head_to_head(self, a: int, b: int) -> tuple[int, int]:
        lo, hi = min(a, b), max(a, b)
        lo_wins, hi_wins = self.h2h.get((lo, hi), (0, 0))
        return (lo_wins, hi_wins) if a == lo else (hi_wins, lo_wins)

    # -- building the model's input row ---------------------------------------

    def side_values(self, pid: int, opponent: int, surface: str) -> dict:
        """Every per-player number the model compares, for one player."""
        p = self.players.loc[pid]
        surface_elo = p[f"elo_{surface.lower()}"]
        values = {
            "elo": p["elo"],
            "surface_elo": surface_elo,
            "blend_elo": (p["elo"] + surface_elo) / 2,
            "log_rank": p["log_rank"],
            "age": p["age"],
            "h2h_wins": self.head_to_head(pid, opponent)[0],
        }
        values.update({f: p[f] for f in HISTORY_FEATURES + STAT_NAMES})
        return values

    def feature_row(self, a: int, b: int, surface: str, best_of: int) -> pd.DataFrame:
        va, vb = self.side_values(a, b, surface), self.side_values(b, a, surface)
        row = {f"diff_{k}": va[k] - vb[k] for k in va}
        wins_a, wins_b = self.head_to_head(a, b)
        row.update({
            "best_of": best_of,
            "min_career_matches": min(self.players.loc[a, "career_matches"], self.players.loc[b, "career_matches"]),
            "min_stats_matches": min(self.players.loc[a, "stats_matches"], self.players.loc[b, "stats_matches"]),
            "h2h_matches": wins_a + wins_b,
        })
        return pd.DataFrame([row])[MODEL_FEATURES].astype(float)

    # -- predicting -----------------------------------------------------------

    def contributions(self, row: pd.DataFrame) -> np.ndarray:
        """SHAP value of each feature for one row (from XGBoost's built-in TreeSHAP)."""
        return self.xgb.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0, :-1]

    def predict(self, a: int, b: int, surface: str, best_of: int = 3, top: int = 6) -> dict:
        """
        Chance that A beats B. We ask the model twice (A vs B, then B vs A) and
        average, so the two players' chances always add up to exactly 100%.
        """
        ab = self.feature_row(a, b, surface, best_of)
        ba = self.feature_row(b, a, surface, best_of)
        p_xgb = (self.xgb.predict_proba(ab)[0, 1] + 1 - self.xgb.predict_proba(ba)[0, 1]) / 2
        p_lr = (self.logreg.predict_proba(ab)[0, 1] + 1 - self.logreg.predict_proba(ba)[0, 1]) / 2
        va, vb = self.side_values(a, b, surface), self.side_values(b, a, surface)
        p_elo = win_probability(va["blend_elo"], vb["blend_elo"])

        impact = (self.contributions(ab) - self.contributions(ba)) / 2
        return {
            "prob_a": float(p_xgb),
            "prob_logreg": float(p_lr),
            "prob_elo": float(p_elo),
            "factors": self.top_factors(impact, va, vb, surface, top),
        }

    def top_factors(self, impact, va, vb, surface, top) -> list[Factor]:
        factors = []
        for i in np.argsort(-np.abs(impact))[:top]:
            feature = MODEL_FEATURES[i]
            a_text, b_text = describe(feature, va, vb, surface)
            factors.append(Factor(feature, FEATURE_LABELS[feature].replace("Surface", surface),
                                  float(impact[i]), a_text, b_text))
        return factors

    def explain_saved_row(self, row: pd.Series, top: int = 6) -> list[Factor]:
        """Top factors for a saved test-set match (the "Replay" tab)."""
        x = row[MODEL_FEATURES].to_frame().T.astype(float)
        impact = self.contributions(x)
        order = np.argsort(-np.abs(impact))[:top]
        return [Factor(MODEL_FEATURES[i], FEATURE_LABELS[MODEL_FEATURES[i]], float(impact[i]),
                       describe_difference(MODEL_FEATURES[i], x.iloc[0, i]), "")
                for i in order]


# ---------------------------------------------------------------------------
# Turning numbers into readable text
# ---------------------------------------------------------------------------

PERCENT_STATS = set(STAT_NAMES)


def fmt_value(name: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if name in PERCENT_STATS:
        return f"{v:.1%}"
    if name == "log_rank":
        rank = np.exp(v)
        return "unranked" if rank >= 1999 else f"#{rank:.0f}"
    if name == "age":
        return f"{v:.1f} yrs"
    if name == "minutes_14d":
        return f"{v:.0f} min"
    return f"{v:.0f}"


def describe(feature: str, va: dict, vb: dict, surface: str) -> tuple[str, str]:
    """Readable values for Player A and Player B."""
    name = feature.removeprefix("diff_")
    if name in va:
        return fmt_value(name, va[name]), fmt_value(name, vb[name])
    return "", ""  # match-level features (best of, experience...) have no per-player value


def describe_difference(feature: str, diff: float) -> str:
    """Readable A-minus-B difference, for saved rows (they store only the differences)."""
    name = feature.removeprefix("diff_")
    if not feature.startswith("diff_"):
        return f"{diff:.0f}"
    if name in PERCENT_STATS:
        return f"{diff * 100:+.1f} pts"
    if name == "log_rank":
        return f"ranking ratio x{np.exp(-diff):.2f}"
    return f"{diff:+.0f}"
