# Phase 2 dataset contract: per-(snapshot, player) kinematics from StatsBomb 360 + DFL tracking

Root: `/hpcwork/p0029161/scoutgpt/_phase2_dataset/` (code in `src/`, outputs in `out/`, verification numbers in `out/verification.json`).
Read-only for consumers. Rebuild: `src/make_split.py` -> `src/build.sbatch` (c25ml, ~15 s) -> `src/verify.sbatch` (c25ml, ~20 s). `build_dataset.py` applies the attack-direction normalisation below; `verify.py` undoes it before comparing with the raw tracking/360 frame and checks the flip itself (goalkeeper side, `atk_flip` consistency).

## Files (`out/`)
| file | content |
|---|---|
| `players/<sb_match_id>.parquet` | one row per (snapshot, player). 56 files, 3,589,806 rows total. |
| `snapshots/<sb_match_id>.parquet` | one row per kept snapshot (163,395 total). Holds context and `row_start`/`n_rows` for window slicing. |
| `splits.csv` | `match_id, dfl_match_id, align_weak, split` (56 rows). Source of truth for the split. |
| `matches.csv` | splits.csv plus snapshot/row counts and per-half alignment offset/median error from `alignment.json`. |
| `build_stats.csv`, `verification.json` | measured build and verification numbers. |

Coordinates: metres, pitch-centre origin, **attack-normalised per snapshot**: the team executing the event (the actor's team) attacks toward +x, in both halves. This is StatsBomb's own native convention (`action_executing_team_left_to_right`). Before normalisation the tables are in the `static_home_left` frame (home attacks +x in both halves, kloppy `STATIC_HOME_AWAY`); a snapshot whose event team is the away team is rotated 180 degrees about the pitch centre, and `atk_flip` records that. The rotation is applied in place, **once per snapshot, uniformly to every column of every row of that snapshot**, including `prev_*`/`next_*` (they come from other snapshots, possibly with the other possession, but are expressed in the current snapshot's frame, never in their own). Rotated columns: `obs_x/y`, `prev_x/y`, `next_x/y`, `lbl_x/y`, `lbl_vx/vy`, `lbl_ax/ay` (negated), `lbl_heading` (shifted by pi, wrapped into (-pi, pi]) and in `snapshots` `actor_x/y`, `lbl_ball_x/y` (negated). Invariant: `lbl_vel`, `lbl_acc`, `lbl_ball_z`, all times and counts. To recover the `static_home_left` frame apply the same operation again wherever `atk_flip` is true (round-trip max deviation 0 for positions; heading is identical modulo 2 pi). 360 `obs_*` and tracking `lbl_*` still share one frame with no further transform (median 360-vs-tracking error 2.19 m, unchanged by the rotation). Measured: median goalkeeper `lbl_x` is -34.8 m (period 1) / -34.0 m (period 2) for the event team and +40.6 / +42.0 m for the opponent, with the correct sign in all 56 matches in both periods; `atk_flip` is true for 81,538 of 163,395 snapshots (49.9 %). Time: `t_s` = `aligned_tracking_time_s`, seconds, period-relative (restarts at each half). Units for velocity m/s, acceleration m/s^2, heading radians.

## Row universe
A `players` row exists for every player that is in the tracking data at the snapshot's aligned frame (`in_tracking`) OR appears in the freeze frame (`observed`).
Typically 22 rows per snapshot (157,260 of 163,395 snapshots); 20 or 21 where tracking has fewer players on the pitch; 23-26 where freeze-frame players are absent from tracking.
Snapshots with no aligned tracking frame (28 of 163,423) are not in the tables, but their observations still feed the timing features.
Ordering: `players` sorted by (`snap_idx`, `object_id`); `snapshots` sorted by `snap_idx`, which orders by (period, `t_s`, StatsBomb `event_index`).

## `players` columns
Inputs (what a model may see) are marked I; labels L; filters/flags F.

| column | dtype | meaning |
|---|---|---|
| `match_id` | int32 | StatsBomb match id. F/key |
| `snap_idx` | int64 | 0-based snapshot order within the match; joins to `snapshots.snap_idx`. key |
| `period` | int8 | 1 or 2. I |
| `t_s` | float64 | aligned tracking time of the snapshot, period-relative seconds. I |
| `object_id` | str | DFL object id (`home_7`, `away_18`), stable within a match. key |
| `sb_player_id` | int64 | StatsBomb player id, stable across matches. I |
| `team_id` | int32 | StatsBomb team id. I |
| `is_home` | bool | home team. I |
| `jersey` | int16 | shirt number. I |
| `pos_id` | Int16 (nullable) | StatsBomb lineup position id at the snapshot time (1=GK ... 24=Left Center Forward); null if the lineup has no on-pitch position then (0.05% of in-tracking rows). I |
| `pos_group` | str (nullable) | GK / DEF (ids 2-8) / MID (9-16, 18-20) / FWD (17, 21-25); my coarse grouping of `pos_id`. I |
| `teammate` | bool | same team as the event actor (defined for every row, also hidden players). I |
| `actor` | bool | this player is the event actor. I |
| `keeper` | bool | `pos_group == "GK"`. Agrees 100% with the 360 `keeper` flag on observed rows. I |
| `in_tracking` | bool | tracking has this player at the aligned frame. F |
| `observed` | bool | player is in this snapshot's freeze frame. I (the unified-task switch) |
| `obs_x`, `obs_y` | float32 | 360 position; NaN iff `observed` is false. I |
| `obs_in_pitch` | boolean (nullable) | 360 `in_pitch`; null when not observed. I |
| `has_prev` | bool | an earlier observation of this player exists in the same period. I |
| `prev_dt` | float32 | seconds from that observation to `t_s`; **-1.0 if `has_prev` is false**. I |
| `prev_x`, `prev_y` | float32 | 360 position at that observation; NaN if `has_prev` is false. I |
| `has_next`, `next_dt`, `next_x`, `next_y` | | same for the first later observation (`next_dt` = seconds from `t_s` to it; -1.0 / NaN if none). I |
| `has_label` | bool | `lbl_x` is non-null (equals `in_tracking` except 0 cases measured). F |
| `lbl_x`, `lbl_y` | float32 | tracked position at the aligned frame. L |
| `lbl_vx`, `lbl_vy`, `lbl_ax`, `lbl_ay` | float32 | tracked velocity/acceleration (Savitzky-Golay, clipped 12 m/s, 6 m/s^2). L |
| `lbl_vel`, `lbl_acc` | float32 | tracked speed and acceleration magnitude. L |
| `lbl_heading` | float32 | `atan2(lbl_vy, lbl_vx)`, radians. Stored. It is the direction of motion; DFL gives no body orientation, so it is noisy at low speed (no mask applied, use `lbl_vel`). L |
| `align_weak` | bool | match-level flag: one of the 8 weak-alignment matches. F |
| `split` | str | train / valid / test (repeated from splits.csv). F |
| `atk_flip` | bool | this snapshot was rotated 180 degrees to put the event team on the +x attacking side (true iff the actor's team is the away team; equals `not is_home` of every `teammate` row; same value on all rows of a snapshot and in `snapshots`). Repeated per row like `split`. I |

Timing semantics (exact): observations are 360 sightings of the same player in snapshots of the same match and period. `prev` is the latest sighting with time strictly before `t_s`, `next` the earliest strictly after. Strictness means a snapshot never "finds itself" and never finds another snapshot at the identical time. The same definition applies to observed and hidden rows. If several snapshots share a time, the sighting from the lowest `event_index` is used. Always mask with `has_prev`/`has_next` before using `prev_dt`, `prev_x`, ...; the -1.0 sentinel is not a time.

Filters to use: model targets and metrics should use `has_label` (equivalently `in_tracking`). Rows with `observed & ~in_tracking` (1,312 rows) have an input but no label.

## `snapshots` columns
`match_id` int32, `snap_idx` int64, `event_id` string, `event_index` int32 (StatsBomb event order), `period` int8, `match_clock_seconds` float32, `t_s` float64, `frame_id` int64 (aligned tracking frame), `offset_s` float32 (alignment offset), `event_type` / `event_sub_type` / `event_outcome` str (from sb_events; sub_type null often), `actor_team_id` int32, `actor_player_id` int64, `actor_is_home` bool, `actor_x`, `actor_y` float32 (the actor's 360 position, which equals the StatsBomb event location; NaN if the actor is not in the freeze frame), `n_visible` int16 (from sb360_frames; equals `n_observed_rows` in all 163,395 snapshots), `n_rows`, `n_in_tracking`, `n_observed_rows`, `n_hidden` (in tracking but not observed), `row_start` (offset of this snapshot's first row in `players/<match>.parquet`), `lbl_ball_x/y/z` float32 and `lbl_ball_status` bool (tracked ball at the aligned frame; label-side, see below), `align_weak`, `split`, `atk_flip` (bool, as in `players`; `actor_is_home == not atk_flip`). `actor_x/y` and `lbl_ball_x/y` are in the attack-normalised frame (see Coordinates); `actor_team_id` was the event-team source, it is assigned for all 163,395 snapshots (0 unassigned, 0 disagreements with the `teammate` rows of `players`).

Deliberately left out of the inputs: event end location (`x_end`, `y_end`) and receiver (they reveal the outcome); `visible_area` (empty `[]` in the match inspected, not checked for all matches); raw `source_x/y`.
Ball: `lbl_ball_*` is not available to a model at inference (only the 360 snapshot is). It is kept as an auxiliary target or diagnostic. The input-side stand-in is the actor position.

## Cutting windows (no second pass, no copies)
For target snapshot `i` in a match and half-width `N`, read the contiguous row range
`lo = snapshots.row_start[i-N]`, `hi = snapshots.row_start[i+N] + snapshots.n_rows[i+N]`, then `players.iloc[lo:hi]` (or a parquet row-group/offset read). Rows keep `snap_idx`, so group by it to get the 2N+1 snapshots; each snapshot's players are keyed by `object_id` for cross-snapshot alignment. Clip `i±N` at the match boundaries. Relative time of snapshot j to the target = `t_s[j] - t_s[i]` (only meaningful within the same `period`; mask or pad across `period` changes). Windows are over 360 snapshots (events that have a freeze frame), not over all StatsBomb events; map to events via `snapshots.event_index`. Tested on 20 random windows (N=5) in each of 56 matches: 0 failures.
Storage if windows were materialised: players parquet is 94.6 MB, so about 1.04 GB at N=5 (x11) and 3.88 GB at N=20 (x41), an arithmetic estimate not a measurement. Slicing needs 0 extra bytes.

## Split
Match-level, 56 matches: train 40 / valid 8 / test 8 (71.4 / 14.3 / 14.3 % of matches). Deterministic: matches ordered by `sha256("phase2-v1:<sb_match_id>")` within the stratum align_weak (6/1/1 of the 8 weak matches) and not weak (34/7/7 of 48). Recorded in `out/splits.csv`; `make_split.py` regenerates it identically. The same teams and players appear in all splits.
