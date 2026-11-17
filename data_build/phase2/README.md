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

`SCHEMA.md` is unchanged, so `imputer/phase2.py` loads the new data without modification; the build itself uses `build_dataset_multi.py` (see below). Its `splits.csv` has the extra columns `dfl_match_id, competition, align_weak`, which `read_splits()` ignores.

### Builder for the multi-competition build

The 330-match dataset is built by `build_dataset_multi.py` and checked by `verify_multi.py`, both copies of the scripts that produced it. `build_dataset.py` and `verify.py` stay untouched as the record of the frozen 56-match build, the same way `make_split.py` and `make_split_multi.py` coexist. The `build.sbatch` and `verify.sbatch` wrappers are left out for the reason given above. `SCHEMA.md` applies to both builders unchanged.

The two multi scripts differ from the frozen ones in five ways:

1. **Conversion root list.** The frozen 56 matches exist only under `_cdfbatch/out`, the new ones under `_cdfbatch_multi/out`. `match_dir()` searches the multi root first and falls back to the original. The five recovered competition-9 matches exist under both, and the multi copy wins. `verify_multi.py` uses the same root list.
2. **Lineup root list.** `lineup_segments()` globs the StatsBomb `raw/lineups` folder of all five competitions (`bundesliga`, `2_bundesliga`, `3_liga`, `frauen_bundesliga`, `dfb_pokal`) instead of the 1. Bundesliga folder alone. Match ids are globally unique, and the newest-version-token rule is unchanged.
3. **`join_status == "both"` filter.** Three matches (4077734, 4085873, 4096284) have 1-2 rows in `lineup_join.parquet` that joined on neither side. Their NaN identities crashed the builder, so rows that did not join are now dropped. `verify_multi.py` applies the same filter. The frozen 56 matches contain only `both` rows, so the filter changes nothing for them.
4. **`JOIN_FIX = {4096284: {"576547": "home_22"}}`.** One hand-verified pairing. StatsBomb player 576547 (Lenni Castrop Choi) wears 26 and DFL object `home_22` (Lenni Castrop) wears 22, so the automatic (team, jersey) join left the player unmatched. The pair was checked by position agreement between the 360 positions and the tracking positions. The builder applies it before the filter in change 3, which asserts that exactly one unjoined row matches. `verify_multi.py` carries the same constant.
5. **Extra time and the match count.** DFB-Pokal match 5003461 has 822 extra-time 360 snapshots (`period` = `first_half_extra` 442, `second_half_extra` 380) with no aligned tracking frame. Snapshots outside the two regulation halves and their observations are dropped and counted in `n_snapshots_no_frame` (`n_snapshots_total` still counts all of them). Lineup segments that start in period 3 or later are ignored, and a segment that starts in regulation and ends in period 3 or later is treated as open-ended. The `build_stats.csv` filename check now expects 330 matches instead of 56, and the split-union check in `verify_multi.py` expects 330 (the result key is `split_union_is_330_files`, formerly `split_union_is_56_files`).

For the 56 frozen 1. Bundesliga matches, `build_dataset_multi.py` reproduces the stored output byte-identically: all 56 `players/` and `snapshots/` parquet files match the original ones by sha256. This was checked twice, independently.

Like the frozen builder, both multi scripts have their input and output paths as constants at the top (`LINEUP_DIRS`, `ROOTS`, `OUT`). `build_dataset_multi.py [--workers N] [sbid ...]` reads `OUT/splits.csv` (written by `make_split_multi.py`).
