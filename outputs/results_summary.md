# Results summary

Test set: **10,382 main-tour matches** from 2023-01-02 to 2026-05-25.
The models learned from 65,600 matches (2000-2021) and used
2,900 matches from 2022 to decide when to stop training.

| Model | Accuracy | Log loss | Brier score |
|---|---|---|---|
| Higher Elo wins | 65.5% | 0.612 | 0.213 |
| Logistic regression | 66.4% | 0.602 | 0.208 |
| Tennis Match Predictor | 66.5% | 0.601 | 0.208 |

*Accuracy* = how often the favourite won. *Log loss* and *Brier score* measure how good
the percentages are (lower is better; always guessing 50% gives 0.693 and 0.250).

## Accuracy by match type (test set)

| Matches | Count | Higher Elo wins | Logistic regression | Tennis Match Predictor |
|---|---|---|---|---|
| All test matches | 10,382 | 65.5% | 66.4% | 66.5% |
| Grand Slams (best of 5) | 1,767 | 70.8% | 72.0% | 72.7% |
| Other tour events | 8,615 | 64.4% | 65.3% | 65.2% |
| Hard courts | 6,022 | 66.3% | 66.7% | 66.7% |
| Clay courts | 3,357 | 64.3% | 65.9% | 66.2% |
| Grass courts | 950 | 64.2% | 66.1% | 66.0% |

## What this means

- **Tennis is hard to predict, and about 2 in 3 is a realistic ceiling.** Every model picks
  the winner roughly 65%-66%
  of the time. Upsets are part of the sport: a best-of-3 match can swing on a handful of points.
- **Elo does most of the work.** A single number per player already gets
  65.5%. Adding serve stats, form, fatigue and head-to-head moves the
  best model (Tennis Match Predictor) to 66.5% accuracy and improves log loss from
  0.612 to 0.601. That is a small, real gain, because
  most of what those features know is already baked into a player's rating.
- **The percentages are trustworthy.** In the calibration chart, when the model says 70%,
  the player wins close to 70% of the time. That matters more than raw accuracy for a
  probability tool.
- **Grand Slams are the most predictable** because best-of-5 gives the better player more
  time to come through.
- **The most influential features** are: Blended Elo rating, ATP ranking, Age, Surface Elo rating, Total points won %.

## Honest limitations

- **Only results and match stats.** The data doesn't know about injuries, illness,
  motivation, weather, coaching changes or whether a player is saving energy for a bigger event.
- **Dates are approximate.** The data records only each tournament's start date, so rest days
  and 14-day fatigue windows are estimates.
- **Serve stats ignore opponent strength.** Winning 70% of serve points against weak
  opponents counts the same as against top players. Elo is the part of the model that
  adjusts for opponent quality.
- **New players are uncertain.** Someone with few matches has a rating that is still settling.
- **The data stops on 2026-06-01.** Predictions in the app use each player's form as
  of that date.
- **The game keeps changing.** The models learned from 2000-2021, and playing styles evolve.
