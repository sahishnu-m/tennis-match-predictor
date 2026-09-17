"""
Phase 5: the web app.

Run it locally:
    streamlit run app.py

Streamlit reruns this whole script from top to bottom every time you click
something. The @st.cache_... decorators make sure the slow parts (loading
the model and data) only happen once.
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loader import APP_DATA_DIR, OUTPUTS_DIR
from src.matchup import SURFACES, Predictor

st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"   # Player A = blue, Player B = orange
MODEL_NAMES = {"elo": "Higher Elo wins", "logreg": "Logistic regression", "xgb": "Tennis Match Predictor"}
ROUND_NAMES = {"F": "Final", "SF": "Semi-final", "QF": "Quarter-final", "R16": "Round of 16",
               "R32": "Round of 32", "R64": "Round of 64", "R128": "Round of 128",
               "RR": "Round robin", "BR": "Bronze medal", "ER": "Early rounds"}


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_csv(name: str, dates: tuple = ()) -> pd.DataFrame:
    return pd.read_csv(APP_DATA_DIR / name, parse_dates=list(dates))


@st.cache_data
def load_metrics() -> dict:
    return json.loads((APP_DATA_DIR / "metrics.json").read_text(encoding="utf-8"))


pred = get_predictor()
players = pred.players
metrics = load_metrics()
data_end = pred.data_end

# Players shown in the dropdowns: anyone with a main-tour match in the last 2 years.
recent = players[players["last_tour_match"] >= data_end - pd.Timedelta(days=730)]
recent = recent.sort_values("elo", ascending=False)


def player_label(pid: int) -> str:
    p = players.loc[pid]
    country = f" ({p['country']})" if isinstance(p["country"], str) else ""
    return f"{p['name']}{country}"


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

def probability_bar(name_a: str, name_b: str, p: float) -> None:
    """A split bar: blue share for Player A, orange share for Player B."""
    pa, pb = round(p * 100, 1), round((1 - p) * 100, 1)
    st.html(f"""
    <div style="font-family:inherit;margin:0.5rem 0 0.25rem">
      <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-end">
        <div style="min-width:0">
          <div style="color:#52514e;font-size:0.9rem">{name_a}</div>
          <div style="color:#0b0b0b;font-size:2.4rem;font-weight:700;line-height:1.1">{pa:.1f}%</div>
        </div>
        <div style="min-width:0;text-align:right">
          <div style="color:#52514e;font-size:0.9rem">{name_b}</div>
          <div style="color:#0b0b0b;font-size:2.4rem;font-weight:700;line-height:1.1">{pb:.1f}%</div>
        </div>
      </div>
      <div role="img" aria-label="{name_a} {pa:.1f}%, {name_b} {pb:.1f}%"
           style="display:flex;gap:2px;height:18px;margin-top:0.5rem">
        <div style="width:{pa}%;background:{COLOR_A};border-radius:4px 0 0 4px"></div>
        <div style="width:{pb}%;background:{COLOR_B};border-radius:0 4px 4px 0"></div>
      </div>
    </div>""")


def factor_chart(factors, name_a: str, name_b: str) -> alt.Chart:
    """Horizontal bars: right = pushes toward Player A, left = toward Player B."""
    df = pd.DataFrame([{
        "Factor": f.label,
        "Impact": f.impact,
        "Favours": name_a if f.impact > 0 else name_b,
        name_a: f.value_a or "-",
        name_b: f.value_b or "-",
    } for f in factors])
    return (
        alt.Chart(df)
        .mark_bar(cornerRadius=4, height=18)
        .encode(
            x=alt.X("Impact:Q", title=f"← {name_b}   |   {name_a} →",
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("Factor:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=240, labelFontSize=12, labelPadding=8)),
            color=alt.Color("Favours:N", scale=alt.Scale(domain=[name_a, name_b], range=[COLOR_A, COLOR_B]),
                            legend=alt.Legend(orient="top", title=None, columnPadding=24,
                                              labelLimit=260, symbolType="square")),
            tooltip=["Factor", "Favours", name_a, name_b],
        )
        .properties(height=alt.Step(30), padding={"left": 12, "right": 8, "top": 4, "bottom": 4})
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Tennis Match Predictor")
st.markdown(
    f"""
This is a sports analytics project that estimates **how likely one ATP player is to beat
another**, using {metrics['sizes']['train'] + metrics['sizes']['validation'] + metrics['sizes']['test']:,}
main-tour matches (plus qualifying and Challenger matches) from 2000 to {data_end:%B %Y}.

It combines **Elo ratings**, **serve and return statistics**, **recent form**, **fatigue** and
**head-to-head records**, and was tested on matches it had never seen.

Data: [Jeff Sackmann's tennis_atp dataset](https://github.com/JeffSackmann/tennis_atp),
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Player numbers are as of **{data_end:%B} {data_end.day}, {data_end.year}**, the last match in the data.
"""
)

# ---------------------------------------------------------------------------
# Part I: the predictor
# ---------------------------------------------------------------------------

st.header("Part I: Pick a matchup", divider="gray")

ids = recent.index.tolist()
default_a = pred.find("Jannik Sinner") if "Jannik Sinner" in set(recent["name"]) else ids[0]
default_b = pred.find("Carlos Alcaraz") if "Carlos Alcaraz" in set(recent["name"]) else ids[1]

col1, col2 = st.columns(2)
with col1:
    a = st.selectbox("Player A", ids, index=ids.index(default_a), format_func=player_label,
                     help="Type to search. Players with a main-tour match in the last 2 years.")
with col2:
    b = st.selectbox("Player B", ids, index=ids.index(default_b), format_func=player_label)

col3, col4 = st.columns([2, 1])
with col3:
    surface = st.segmented_control("Surface", SURFACES, default="Hard") or "Hard"
with col4:
    best_of_5 = st.toggle("Best of 5 sets", help="Grand Slam format. Longer matches favour the stronger player.")

if a == b:
    st.info("Pick two different players.")
else:
    name_a, name_b = players.loc[a, "name"], players.loc[b, "name"]
    result = pred.predict(a, b, surface, 5 if best_of_5 else 3)
    p = result["prob_a"]

    probability_bar(name_a, name_b, p)
    st.caption(
        f"Tennis Match Predictor on {surface.lower()} courts, best of {5 if best_of_5 else 3}. "
        f"For comparison, {name_a}'s chance from logistic regression is {result['prob_logreg']:.0%}, "
        f"and from Elo ratings alone {result['prob_elo']:.0%}."
    )

    wins_a, wins_b = pred.head_to_head(a, b)
    if wins_a + wins_b:
        st.markdown(f"**Head-to-head:** {name_a} {wins_a}–{wins_b} {name_b} (all surfaces, since 2000)")
    else:
        st.markdown("**Head-to-head:** no previous meetings in the data")

    st.subheader("What's driving this prediction")
    st.altair_chart(factor_chart(result["factors"], name_a, name_b), width="stretch")
    st.caption("Bar length shows how much each factor moved the prediction (SHAP values). "
               "Hover a bar to see both players' numbers.")

    with st.expander("Compare the two players side by side"):
        pa, pb = players.loc[a], players.loc[b]
        rows = [
            ("Elo rating (overall)", f"{pa['elo']:.0f}", f"{pb['elo']:.0f}"),
            (f"Elo rating ({surface})", f"{pa['elo_' + surface.lower()]:.0f}", f"{pb['elo_' + surface.lower()]:.0f}"),
            ("ATP ranking", *(f"#{r:.0f}" if pd.notna(r) else "unranked" for r in (pa["rank"], pb["rank"]))),
            ("Age", f"{pa['age']:.1f}", f"{pb['age']:.1f}"),
            ("Wins in last 10 matches", f"{pa['wins_last10']:.0f}", f"{pb['wins_last10']:.0f}"),
            ("Days since last match", f"{pa['days_since_last']:.0f}", f"{pb['days_since_last']:.0f}"),
            ("1st serve in", f"{pa['first_serve_in']:.1%}", f"{pb['first_serve_in']:.1%}"),
            ("1st serve points won", f"{pa['first_serve_won']:.1%}", f"{pb['first_serve_won']:.1%}"),
            ("2nd serve points won", f"{pa['second_serve_won']:.1%}", f"{pb['second_serve_won']:.1%}"),
            ("Aces per serve point", f"{pa['ace_rate']:.1%}", f"{pb['ace_rate']:.1%}"),
            ("Double faults per serve point", f"{pa['double_fault_rate']:.1%}", f"{pb['double_fault_rate']:.1%}"),
            ("Break points saved", f"{pa['bp_saved']:.1%}", f"{pb['bp_saved']:.1%}"),
            ("Return points won", f"{pa['return_points_won']:.1%}", f"{pb['return_points_won']:.1%}"),
            ("Total points won", f"{pa['total_points_won']:.1%}", f"{pb['total_points_won']:.1%}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Stat", name_a, name_b]),
                     hide_index=True, width="stretch")
        st.caption("Serve and return stats cover each player's last 20 matches with stats.")

# ---------------------------------------------------------------------------
# Part II: accuracy
# ---------------------------------------------------------------------------

st.header("Part II: How accurate is this?", divider="gray")
test = metrics["test"]
st.markdown(
    f"""
The models learned from matches between 2000 and 2021, and then were tested on
**{metrics['sizes']['test']:,} main-tour matches from {metrics['test_period']}**
that they had never seen. The simplest approach, "the player with the higher Elo rating wins",
is the baseline to beat.
"""
)
cols = st.columns(3)
for col, key in zip(cols, MODEL_NAMES):
    delta = None if key == "elo" else f"{(test[key]['accuracy'] - test['elo']['accuracy']) * 100:+.1f} pts vs Elo"
    col.metric(MODEL_NAMES[key], f"{test[key]['accuracy']:.1%}", delta, help="Share of test matches where the favourite won")

st.image(str(OUTPUTS_DIR / "model_comparison.png"), width="stretch")

slices = pd.DataFrame(metrics["slices"]).rename(columns={"group": "Matches", "matches": "Count"})
st.dataframe(
    slices, hide_index=True, width="stretch",
    column_config={k: st.column_config.ProgressColumn(v, format="percent", min_value=0.5, max_value=0.8)
                   for k, v in MODEL_NAMES.items()},
)
st.caption("Bars start at 50% (a coin flip). Grand Slams are best of 5 sets, which makes upsets rarer.")

st.subheader("Can you trust the percentages?")
st.markdown(
    "Accuracy only asks *who* was favoured. **Calibration** asks whether a 70% prediction "
    "really comes true about 70% of the time. Points close to the dashed line mean the "
    "percentages can be taken at face value."
)
st.image(str(OUTPUTS_DIR / "calibration.png"), width="stretch")

# ---------------------------------------------------------------------------
# Part III: what matters
# ---------------------------------------------------------------------------

st.header("Part III: What decides tennis matches?", divider="gray")
imp = load_csv("feature_importance.csv")
groups = imp.groupby("group", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False)
groups["share"] = groups["mean_abs_shap"] / groups["mean_abs_shap"].sum()

st.markdown(
    """
Each prediction can be split into how much every feature pushed it one way or the other
(these are called **SHAP values**). Averaging those pushes over thousands of test matches
shows what the model relies on most.
"""
)
st.dataframe(
    groups[["group", "share"]], hide_index=True, width="stretch",
    column_config={
        "group": "Type of information",
        "share": st.column_config.ProgressColumn("Share of the model's attention", format="percent",
                                                 min_value=0, max_value=1),
    },
)
top_group_share = groups["share"].iloc[0]
st.markdown(
    f"""
- **Player strength is everything.** Elo ratings and ranking account for about
  {top_group_share:.0%} of the model's attention. Who has been beating whom is the best summary
  of how good someone is.
- **Serve and return stats add detail.** The best single one is *total points won*: a player
  who wins a higher share of all points is usually better than their record suggests.
- **Rust and fatigue matter a little.** Long layoffs and heavy recent schedules nudge
  predictions slightly. Skill matters far more.
- **Head-to-head matters less than people think.** Once you know how good both players are,
  their past meetings add surprisingly little.
"""
)
with st.expander("See every feature"):
    st.image(str(OUTPUTS_DIR / "feature_importance.png"), width="stretch")
    st.dataframe(
        imp[["label", "group", "mean_abs_shap"]], hide_index=True, width="stretch",
        column_config={
            "label": "Feature", "group": "Type",
            "mean_abs_shap": st.column_config.ProgressColumn("Average impact", format="%.3f",
                                                             min_value=0, max_value=float(imp["mean_abs_shap"].max())),
        },
    )

# ---------------------------------------------------------------------------
# Part IV: replay
# ---------------------------------------------------------------------------

st.header("Part IV: Replay a past match", divider="gray")
st.markdown(
    "Pick a main-tour match from 2023 onward and see what the model predicted **before** it "
    "was played, using only information available at the time."
)
tests = load_csv("test_predictions.csv", dates=("date",))
tests["year"] = tests["date"].dt.year

c1, c2, c3 = st.columns([1, 2, 1.3])
with c1:
    year = st.selectbox("Year", sorted(tests["year"].unique(), reverse=True))
in_year = tests[tests["year"] == year]
with c2:
    events = in_year.groupby("tourney_name")["tourney_level"].first()
    order = sorted(events.index, key=lambda t: (events[t] != "G", events[t] != "M", t))
    event = st.selectbox("Tournament", order, help="Grand Slams first, then Masters 1000, then the rest.")
in_event = in_year[in_year["tourney_name"] == event]
with c3:
    rounds = [r for r in ROUND_NAMES if r in set(in_event["round"])]
    rnd = st.selectbox("Round", rounds, format_func=lambda r: ROUND_NAMES[r])

matches = in_event[in_event["round"] == rnd]
labels = {i: f"{m.player_a} vs {m.player_b}" for i, m in matches.iterrows()}
pick = st.selectbox("Match", list(labels), format_func=labels.get)
m = tests.loc[pick]

winner, loser = (m["player_a"], m["player_b"]) if m["a_won"] else (m["player_b"], m["player_a"])
p_winner = m["prob_xgb"] if m["a_won"] else 1 - m["prob_xgb"]
probability_bar(m["player_a"], m["player_b"], m["prob_xgb"])
if p_winner > 0.5:
    st.success(f"**{winner}** won {m['score']}. The model favoured them ({p_winner:.0%}). ✓")
else:
    st.warning(f"**{winner}** won {m['score']}, an upset by the model's numbers: it gave them only {p_winner:.0%}.")
factors = pred.explain_saved_row(m)
st.altair_chart(factor_chart(factors, m["player_a"], m["player_b"]), width="stretch")
st.caption(f"{m['surface']} court, best of {m['best_of']}. Tooltip values show {m['player_a']} minus {m['player_b']}.")

# ---------------------------------------------------------------------------
# Part V: browse ratings
# ---------------------------------------------------------------------------

st.header("Part V: Browse player ratings", divider="gray")
active_days = st.slider("Only players with a main-tour match in the last … days", 30, 730, 365, step=30)
table = players[players["last_tour_match"] >= data_end - pd.Timedelta(days=active_days)]
table = table.sort_values("elo", ascending=False).head(200)
elo_min, elo_max = float(table["elo"].min()), float(table["elo"].max())
elo_cfg = lambda label: st.column_config.ProgressColumn(label, format="%.0f", min_value=elo_min - 50, max_value=elo_max)
st.dataframe(
    table[["name", "country", "rank", "elo", "elo_hard", "elo_clay", "elo_grass", "wins_last10", "total_points_won"]],
    hide_index=True, width="stretch", height=420,
    column_config={
        "name": "Player", "country": "Country",
        "rank": st.column_config.NumberColumn("ATP rank", format="%d"),
        "elo": elo_cfg("Elo"), "elo_hard": elo_cfg("Hard"), "elo_clay": elo_cfg("Clay"), "elo_grass": elo_cfg("Grass"),
        "wins_last10": st.column_config.NumberColumn("Last 10 W", format="%d"),
        "total_points_won": st.column_config.NumberColumn("Points won", format="percent"),
    },
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built as a student sports analytics project. Predictions are estimates based on past results "
    "and cannot account for injuries, conditions or anything else outside the data. "
    "Match data © Jeff Sackmann / Tennis Abstract, "
    "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)."
)
