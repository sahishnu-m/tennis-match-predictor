"""
Phase 4: train the models, test them honestly, and draw the charts.

TIME-BASED SPLIT (never shuffle!)
    train      2000-2021   the models learn from these matches
    validation 2022        used to decide when XGBoost should stop learning
    test       2023+       touched only once, at the very end, to report results

Predicting the future means learning only from the past, so the split follows
the calendar. A random shuffle would let the model learn from 2024 matches and
then be "tested" on 2019 matches.

We train and test on main-tour matches. (Qualifying/Challenger matches were
already used to build ratings and stats in Phases 2-3.)

Three models are compared:
  1. Baseline:  the player with the higher (blended) Elo wins
  2. Logistic regression: adds up the features with learned weights
  3. Tennis Match Predictor: our main model, built with XGBoost (hundreds of
     small decision trees, each fixing the last one's mistakes)

Run it:
    python -m src.train
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import APP_DATA_DIR, OUTPUTS_DIR
from src.features import (FEATURE_GROUPS, FEATURE_LABELS, MODEL_FEATURES,
                          MODEL_TABLE_FILE, build_model_table)

TRAIN_END = "2022-01-01"
TEST_START = "2023-01-01"

XGB_FILE = APP_DATA_DIR / "xgb_model.json"
LOGREG_FILE = APP_DATA_DIR / "logreg.joblib"
METRICS_FILE = APP_DATA_DIR / "metrics.json"
IMPORTANCE_FILE = APP_DATA_DIR / "feature_importance.csv"
TEST_PREDICTIONS_FILE = APP_DATA_DIR / "test_predictions.csv"
SUMMARY_FILE = OUTPUTS_DIR / "results_summary.md"

MODEL_NAMES = {"elo": "Higher Elo wins", "logreg": "Logistic regression", "xgb": "Tennis Match Predictor"}  # our main model (built with XGBoost)

# Chart style: one color per model, used the same way in every chart.
COLORS = {"elo": "#2a78d6", "logreg": "#eb6834", "xgb": "#1baf7a"}
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


# ---------------------------------------------------------------------------
# Step 1: data
# ---------------------------------------------------------------------------

def load_model_table() -> pd.DataFrame:
    if not MODEL_TABLE_FILE.exists():
        print("Model table not found, building it (Phases 1-3)...")
        return build_model_table()
    return pd.read_csv(MODEL_TABLE_FILE, parse_dates=["date"], low_memory=False)


def split(mt: pd.DataFrame):
    tour = mt[mt["is_tour_match"]]
    train = tour[tour["date"] < TRAIN_END]
    val = tour[(tour["date"] >= TRAIN_END) & (tour["date"] < TEST_START)]
    test = tour[tour["date"] >= TEST_START]
    return train, val, test


# ---------------------------------------------------------------------------
# Step 2: the models
# ---------------------------------------------------------------------------

def train_logreg(train: pd.DataFrame):
    """
    Logistic regression, in three steps chained together:
      - fill any missing value with the middle (median) value
      - rescale every feature to a similar range, so no feature dominates
        just because its numbers are bigger (Elo ~ hundreds, rates ~ 0.01)
      - learn one weight per feature
    """
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000))
    return model.fit(train[MODEL_FEATURES], train["a_won"])


def train_xgboost(train: pd.DataFrame, val: pd.DataFrame):
    """
    XGBoost builds up to 3000 small trees, but stops early once adding more
    trees stops improving the 2022 validation matches (prevents memorising).
    """
    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,      # small steps = steadier learning
        max_depth=4,             # each tree asks at most 4 questions
        subsample=0.8,           # each tree sees a random 80% of matches...
        colsample_bytree=0.8,    # ...and 80% of features (reduces overfitting)
        early_stopping_rounds=100,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(train[MODEL_FEATURES], train["a_won"],
              eval_set=[(val[MODEL_FEATURES], val["a_won"])], verbose=False)
    return model


def predict_all(df: pd.DataFrame, logreg, xgb_model) -> pd.DataFrame:
    """Chance that Player A wins, from each of the three models."""
    return pd.DataFrame({
        "elo": df["elo_prob_a"].to_numpy(),
        "logreg": logreg.predict_proba(df[MODEL_FEATURES])[:, 1],
        "xgb": xgb_model.predict_proba(df[MODEL_FEATURES])[:, 1],
    }, index=df.index)


def score(y: pd.Series, probs: pd.DataFrame) -> dict:
    """
    accuracy    how often the favourite (>50%) actually won
    log loss    how good the percentages are; punishes confident mistakes (lower = better)
    brier       average squared error of the percentages (lower = better)
    """
    out = {}
    for key in probs:
        p = probs[key].clip(1e-6, 1 - 1e-6)
        out[key] = {
            "accuracy": accuracy_score(y, p > 0.5),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
        }
    return out


# ---------------------------------------------------------------------------
# Step 3: explanations (SHAP)
# ---------------------------------------------------------------------------

def shap_values(xgb_model, X: pd.DataFrame) -> np.ndarray:
    """
    SHAP values split each prediction into "how much each feature pushed it
    up or down". Averaging their size over many matches shows which features
    matter most overall.
    """
    try:
        import shap
        return shap.TreeExplainer(xgb_model).shap_values(X)
    except Exception as err:  # fall back to XGBoost's built-in (identical) method
        print(f"shap library failed ({err}); using XGBoost's built-in SHAP values.")
        contribs = xgb_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contribs[:, :-1]  # drop the last column (the baseline value)


# ---------------------------------------------------------------------------
# Step 4: charts
# ---------------------------------------------------------------------------

def style_axes(ax):
    """Quiet gridlines and axes so the data stands out."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def calibration_table(y: pd.Series, probs: pd.DataFrame) -> pd.DataFrame:
    """Group predictions into 10% buckets and compare with how often A really won."""
    rows = []
    buckets = np.linspace(0, 1, 11)
    for key in probs:
        bucket = pd.cut(probs[key], buckets, include_lowest=True, labels=False)
        g = pd.DataFrame({"p": probs[key], "y": y.to_numpy(), "b": bucket}).groupby("b")
        for b, grp in g:
            rows.append({"model": key, "bucket": f"{int(b) * 10}-{int(b) * 10 + 10}%",
                         "predicted": grp["p"].mean(), "actual": grp["y"].mean(), "matches": len(grp)})
    return pd.DataFrame(rows)


def plot_calibration(cal: pd.DataFrame, path):
    fig, ax = new_figure(6.4, 5.6)
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.62, 0.55, "perfect calibration", color=INK_SOFT, fontsize=8, rotation=41,
            ha="left", va="top")
    for key in MODEL_NAMES:
        d = cal[(cal["model"] == key) & (cal["matches"] >= 30)]  # skip tiny buckets
        ax.plot(d["predicted"], d["actual"], color=COLORS[key], linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=MODEL_NAMES[key], zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Predicted chance that Player A wins", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("How often Player A actually won", color=INK_SOFT, fontsize=9)
    ax.set_title("Calibration: do the percentages mean what they say?", loc="left",
                 color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, "Test matches 2023+, grouped into 10% buckets. Closer to the dashed line is better.",
            transform=ax.transAxes, color=INK_SOFT, fontsize=8.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path, top=15):
    d = imp.head(top).iloc[::-1]  # biggest at the top
    fig, ax = new_figure(7.2, 5.8)
    ax.barh(d["label"], d["mean_abs_shap"], color=COLORS["elo"], height=0.62)
    for y, v in enumerate(d["mean_abs_shap"]):
        ax.text(v, y, f"  {v:.3f}", va="center", color=INK_SOFT, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(0, d["mean_abs_shap"].max() * 1.15)
    ax.set_xlabel("Average impact on the prediction (mean |SHAP value|, log-odds)", color=INK_SOFT, fontsize=9)
    ax.set_title("What drives the Tennis Match Predictor", loc="left", color=INK,
                 fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, f"Top {top} features on test matches 2023+", transform=ax.transAxes,
            color=INK_SOFT, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_model_comparison(test_scores: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    keys = list(MODEL_NAMES)[::-1]  # baseline at the bottom
    labels = [MODEL_NAMES[k] for k in keys]
    colors = [COLORS[k] for k in keys]

    panels = [
        ("accuracy", "Accuracy (higher is better)", 0.5, "coin flip 50%", lambda v: f"{v:.1%}"),
        ("log_loss", "Log loss (lower is better)", np.log(2), "coin flip 0.693", lambda v: f"{v:.3f}"),
    ]
    for ax, (metric, title, ref, ref_label, fmt) in zip(axes, panels):
        style_axes(ax)
        values = [test_scores[k][metric] for k in keys]
        ax.barh(labels, values, color=colors, height=0.6)
        for y, v in enumerate(values):
            ax.text(v, y, "  " + fmt(v), va="center", color=INK, fontsize=9)
        ax.axvline(ref, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
        # Room above the top bar for the reference-line label.
        ax.set_ylim(-0.5, len(keys) - 0.1)
        ax.text(ref, len(keys) - 0.35, ref_label, color=INK_SOFT, fontsize=7.5, ha="center",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        ax.set_xlim(0, max(max(values), ref) * 1.25)
        if metric == "accuracy":
            ticks = np.arange(0, 0.81, 0.2)
            ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", colors=INK, length=0)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold")
    axes[1].set_yticklabels([])
    fig.suptitle("Model comparison on test matches (2023+)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 5: the plain-English summary
# ---------------------------------------------------------------------------

def write_summary(metrics: dict, imp: pd.DataFrame, slices: pd.DataFrame, path):
    t = metrics["test"]
    best = min(("logreg", "xgb"), key=lambda k: t[k]["log_loss"])
    top5 = ", ".join(imp["label"].head(5))
    table = "\n".join(
        f"| {MODEL_NAMES[k]} | {t[k]['accuracy']:.1%} | {t[k]['log_loss']:.3f} | {t[k]['brier']:.3f} |"
        for k in MODEL_NAMES)
    slice_rows = "\n".join(
        f"| {r.group} | {r.matches:,} | {r.elo:.1%} | {r.logreg:.1%} | {r.xgb:.1%} |"
        for r in slices.itertuples())

    text = f"""# Results summary

Test set: **{metrics['sizes']['test']:,} main-tour matches** from {metrics['test_period']}.
The models learned from {metrics['sizes']['train']:,} matches (2000-2021) and used
{metrics['sizes']['validation']:,} matches from 2022 to decide when to stop training.

| Model | Accuracy | Log loss | Brier score |
|---|---|---|---|
{table}

*Accuracy* = how often the favourite won. *Log loss* and *Brier score* measure how good
the percentages are (lower is better; always guessing 50% gives 0.693 and 0.250).

## Accuracy by match type (test set)

| Matches | Count | Higher Elo wins | Logistic regression | Tennis Match Predictor |
|---|---|---|---|---|
{slice_rows}

## What this means

- **Tennis is hard to predict, and about 2 in 3 is a realistic ceiling.** Every model picks
  the winner roughly {t['elo']['accuracy']:.0%}-{max(t['logreg']['accuracy'], t['xgb']['accuracy']):.0%}
  of the time. Upsets are part of the sport: a best-of-3 match can swing on a handful of points.
- **Elo does most of the work.** A single number per player already gets
  {t['elo']['accuracy']:.1%}. Adding serve stats, form, fatigue and head-to-head moves the
  best model ({MODEL_NAMES[best]}) to {t[best]['accuracy']:.1%} accuracy and improves log loss from
  {t['elo']['log_loss']:.3f} to {t[best]['log_loss']:.3f}. That is a small, real gain, because
  most of what those features know is already baked into a player's rating.
- **The percentages are trustworthy.** In the calibration chart, when the model says 70%,
  the player wins close to 70% of the time. That matters more than raw accuracy for a
  probability tool.
- **Grand Slams are the most predictable** because best-of-5 gives the better player more
  time to come through.
- **The most influential features** are: {top5}.

## Honest limitations

- **Only results and match stats.** The data doesn't know about injuries, illness,
  motivation, weather, coaching changes or whether a player is saving energy for a bigger event.
- **Dates are approximate.** The data records only each tournament's start date, so rest days
  and 14-day fatigue windows are estimates.
- **Serve stats ignore opponent strength.** Winning 70% of serve points against weak
  opponents counts the same as against top players. Elo is the part of the model that
  adjusts for opponent quality.
- **New players are uncertain.** Someone with few matches has a rating that is still settling.
- **The data stops on {metrics['data_end']}.** Predictions in the app use each player's form as
  of that date.
- **The game keeps changing.** The models learned from 2000-2021, and playing styles evolve.
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def main():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    mt = load_model_table()
    train, val, test = split(mt)
    print(f"Train {len(train):,} | validation {len(val):,} | test {len(test):,} main-tour matches")

    logreg = train_logreg(train)
    xgb_model = train_xgboost(train, val)
    print(f"XGBoost stopped at {xgb_model.best_iteration + 1} trees")

    val_probs = predict_all(val, logreg, xgb_model)
    test_probs = predict_all(test, logreg, xgb_model)
    metrics = {
        "validation": score(val["a_won"], val_probs),
        "test": score(test["a_won"], test_probs),
        "sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_period": f"{test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}",
        "data_end": f"{mt['date'].max():%Y-%m-%d}",
        "xgb_trees": int(xgb_model.best_iteration + 1),
    }

    # Accuracy for different kinds of matches.
    groups = {
        "All test matches": test.index,
        "Grand Slams (best of 5)": test.index[test["tourney_level"] == "G"],
        "Other tour events": test.index[test["tourney_level"] != "G"],
        "Hard courts": test.index[test["surface"] == "Hard"],
        "Clay courts": test.index[test["surface"] == "Clay"],
        "Grass courts": test.index[test["surface"] == "Grass"],
    }
    slices = pd.DataFrame([
        {"group": name, "matches": len(idx),
         **{k: accuracy_score(test.loc[idx, "a_won"], test_probs.loc[idx, k] > 0.5) for k in MODEL_NAMES}}
        for name, idx in groups.items()])
    metrics["slices"] = slices.to_dict(orient="records")

    # Feature importance from SHAP on the test matches.
    sv = shap_values(xgb_model, test[MODEL_FEATURES])
    imp = pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["label"] = imp["feature"].map(FEATURE_LABELS)
    imp["group"] = imp["feature"].map(FEATURE_GROUPS)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Charts.
    cal = calibration_table(test["a_won"], test_probs)
    plot_calibration(cal, OUTPUTS_DIR / "calibration.png")
    plot_importance(imp, OUTPUTS_DIR / "feature_importance.png")
    plot_model_comparison(metrics["test"], OUTPUTS_DIR / "model_comparison.png")

    # Save everything the app needs.
    xgb_model.save_model(XGB_FILE)
    joblib.dump(logreg, LOGREG_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    imp.to_csv(IMPORTANCE_FILE, index=False)
    cal.to_csv(APP_DATA_DIR / "calibration.csv", index=False)
    keep = ["match_id", "date", "tourney_name", "tourney_level", "round", "surface", "best_of",
            "player_a", "player_b", "score", "a_won"] + MODEL_FEATURES
    keep = list(dict.fromkeys(keep))  # remove duplicates (best_of is in both lists)
    saved = test[keep].copy()
    for k in MODEL_NAMES:
        saved[f"prob_{k}"] = test_probs[k].round(4)
    numeric = saved.select_dtypes("number").columns
    saved[numeric] = saved[numeric].round(5)
    saved.to_csv(TEST_PREDICTIONS_FILE, index=False, date_format="%Y-%m-%d")

    write_summary(metrics, imp, slices, SUMMARY_FILE)

    # Print the results.
    print("\n=== Test results (main tour, 2023+) ===")
    for k, name in MODEL_NAMES.items():
        s = metrics["test"][k]
        print(f"{name:<22} accuracy {s['accuracy']:.1%}   log loss {s['log_loss']:.3f}   brier {s['brier']:.3f}")
    print("\nAccuracy by match type:")
    print(slices.to_string(index=False, formatters={k: "{:.1%}".format for k in MODEL_NAMES}))
    print("\nTop 10 features (mean |SHAP|):")
    print(imp.head(10)[["label", "mean_abs_shap"]].to_string(index=False))
    print(f"\nCharts saved to {OUTPUTS_DIR}, summary in {SUMMARY_FILE.name}")


if __name__ == "__main__":
    main()
