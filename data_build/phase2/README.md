# Phase 2 dataset builder (reference copy)

These files are unchanged copies of `_phase2_dataset/src/` and `SCHEMA.md` from `phase2_dataset.tgz`. That archive is the build of the 56 1. Bundesliga 2026/27 matches that `imputer/phase2.py` loads.

- `SCHEMA.md` is the column contract for `players/` and `snapshots/`. A new build has to keep it exactly so that `imputer/phase2.py` loads the new data unchanged.
- `build_dataset.py` reads per-match upstream files from `_cdfbatch/out/<season>/<sb_match_id>_<dfl_match_id>/`. Those files are produced by a separate 360 ↔ tracking alignment step that is not in this repo:
  - `sb360_frames.parquet`, `sb360_objects.parquet`, `sb_events.parquet`
  - `lineup_join.parquet`, `tracking.parquet`
  - the StatsBomb lineups JSON
- `make_split.py` writes the train/valid/test split (match level, hash ordered, stratified by `align_weak`). `verify.py` checks the build against the raw tracking and 360 data.
- The paths (`/hpcwork/p0029161/...`) are those of the CLAIX cluster the dataset was built on. To build anywhere else, change the path constants. The `build.sbatch` and `verify.sbatch` job wrappers are left out because they contain a personal cluster user path. Each one runs `build_dataset.py --workers 16` or `verify.py` with Python 3.11.

No data is stored in this repository.

## Multi-competition build

`make_split_multi.py` builds a dataset over five competitions: 1. Bundesliga (9), 2. Bundesliga (10), Frauen-Bundesliga (135), 3. Liga (179) and DFB-Pokal (165). It supersedes `make_split.py` for this build. `make_split.py` remains the record of the frozen 56-match 1. Bundesliga build, whose split values `make_split_multi.py` carries over unchanged (its 8-match test set has been evaluated once and must not move). Additional 1. Bundesliga matches are all `train`.

Matches are selected in this order, using the conversion output of each match (`alignment.json`, `sb360_objects.parquet`):

1. `player_id` fill rate of `sb360_objects.parquet` >= 0.5. Anonymous 360 cannot feed an identity-based model.
2. No period's `flags` contain `BOUNDARY`, `NOT_V_SHAPED` or `AMBIGUOUS_MINIMUM`: the time offset was not identified.
3. Worst-half `median_m` <= 8.0 m. Alignment error corrupts the labels directly.
4. `align_weak = True` when the worst-half `median_m` > 4.12 m (the threshold of the 56-match build). Such matches are still used for training; the flag only drives stratification.

The measured counts behind each threshold are in the comments of the script, and the per-competition counts and drop reasons are written to `splits_summary.json`. The split is hash ordered (key `phase3-<comp>-v1:<sb_match_id>`) and stratified by `align_weak`: 70/15/15 for competitions 10 and 135, and train 0 / valid 15% / test 85% for competitions 179 and 165, which are held out of training.

`build_dataset.py` and `SCHEMA.md` are unchanged, so `imputer/phase2.py` loads the new data without modification. Its `splits.csv` has the extra columns `dfl_match_id, competition, align_weak`, which `read_splits()` ignores.
