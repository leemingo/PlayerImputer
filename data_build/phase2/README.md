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
