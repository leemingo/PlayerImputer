# Phase 5: cross-league generalisation

## Question

How does a model trained on the 1. Bundesliga alone perform on other German competitions, and how does that change when 2. Bundesliga and Frauen-Bundesliga are added to training?

## Data

`data_build/phase2/make_split_multi.py` selects the usable matches and writes `splits.csv` (columns `match_id, dfl_match_id, competition, align_weak, split`) and `splits_summary.json`. The selection and split rules are in that script and in `data_build/phase2/README.md`. The dataset layout is unchanged (`data_build/phase2/SCHEMA.md`), so `imputer/phase2.py` loads it without modification. Competition codes: 9 = 1. Bundesliga, 10 = 2. Bundesliga, 135 = Frauen-Bundesliga, 179 = 3. Liga, 165 = DFB-Pokal.

The 1. Bundesliga test set is 8 matches and frozen. It comes from the 56-match build, and the published baseline below was measured on it once. `make_split_multi.py` carries those 56 assignments over unchanged and asserts it. Do not resplit it.

## Training regimes

- A: 1. Bundesliga train only.
- B: 1. Bundesliga + 2. Bundesliga + Frauen-Bundesliga train.

Both regimes are evaluated on the same five test sets (one per competition), so a difference between A and B comes from the training data and not from different evaluation rows.

3. Liga and DFB-Pokal are never trained on (train share 0, their train share goes to test). They measure generalisation to a league the model has not seen in either regime. If B improves there, mixing leagues helps transfer in general. If it does not, a league has to be in training to benefit. A and B alone cannot separate these two outcomes. DFB-Pokal also mixes teams from three divisions, which would blur "league level" as a variable if it were trained on.

## Evaluation

`Phase2Windows` takes `match_ids`. Per-competition evaluation needs no code change: filter `splits.csv` by `competition` and `split` and pass that list.

Normalisation statistics must come from the training run's `phase2_stats.json`. Refitting them on an evaluation competition is leakage.

## Baseline to beat

CPU model, 56-match 1. Bundesliga build, test set evaluated once:

| Metric | Value |
|---|---|
| Visible-player velocity-vector RMSE | 1.890 m/s |
| Hidden-player position median error | 4.088 m |

Comparability caveat: 58.8% of visible rows share `(match, period, t_s, player)` with another row, so these baseline metrics are snapshot-weighted. A model must be scored on the same row set, otherwise the numbers are not comparable.
