"""Independent verification of out/. Prints a report and writes out/verification.json."""
import json, os, glob, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOTS = ["/hpcwork/p0029161/scoutgpt/_cdfbatch_multi/out/2026-2027",   # competitions 10/135/165/179 + new/recovered 9
         "/hpcwork/p0029161/scoutgpt/_cdfbatch/out/2026-2027"]         # the frozen 56 (competition 9) live only here
OUT = "/hpcwork/p0029161/scoutgpt/_phase2_multileague/out"

JOIN_FIX = {4096284: {"576547": "home_22"}}


def match_dir(sbid, dfl):
    import os
    for r in ROOTS:
        if os.path.isdir(f"{r}/{sbid}_{dfl}"):
            return f"{r}/{sbid}_{dfl}"
    raise FileNotFoundError(f"{sbid}_{dfl}")
LBL = ["x", "y", "vx", "vy", "ax", "ay", "vel", "acc"]
SPATIAL = ["obs_x", "obs_y", "prev_x", "prev_y", "next_x", "next_y", "lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay"]
rng_seed = 0


def one(args):
    sbid, dfl = args
    d = match_dir(sbid, dfl)
    p = pd.read_parquet(f"{OUT}/players/{sbid}.parquet")
    s = pd.read_parquet(f"{OUT}/snapshots/{sbid}.parquet")
    fr = pd.read_parquet(f"{d}/sb360_frames.parquet"); ob = pd.read_parquet(f"{d}/sb360_objects.parquet")
    lj = pd.read_parquet(f"{d}/lineup_join.parquet")
    lj = lj[lj.join_status == "both"]
    for sbp, obj in JOIN_FIX.get(sbid, {}).items(): lj = pd.concat([lj, pd.DataFrame({"sb_player_id": [sbp], "dfl_object_id": [obj]})])   # same pair as build_dataset.py
    tr = pd.read_parquet(f"{d}/tracking.parquet")
    r = {"match_id": sbid}
    # the tables are attack-normalised (atk_flip); the checks below compare with the RAW tracking/360 frame, so undo the rotation first
    p = p.copy(); f = p.atk_flip.to_numpy()
    for c in SPATIAL: p[c] = np.where(f, -p[c], p[c]).astype("float32")
    # (1) brief's join recipe, independently, vs our labels for observed rows
    sb2obj = dict(zip(lj.sb_player_id.astype(str), lj.dfl_object_id.astype(str)))
    o = ob.merge(fr[["event_id", "aligned_tracking_frame_id", "period", "match_clock_seconds"]], on="event_id", how="left")
    o["obj"] = o.player_id.astype(str).map(sb2obj)
    tr2 = tr[tr.ball == False][["frame_id", "object_id", "x", "y", "vx", "vy", "ax", "ay", "vel", "acc"]].copy()
    tr2["object_id"] = tr2.object_id.astype(str)
    tr2 = tr2.rename(columns={c: c + "_t" for c in LBL})
    m = o.merge(tr2, left_on=["aligned_tracking_frame_id", "obj"], right_on=["frame_id", "object_id"], how="left")
    m = m.merge(s[["event_id", "snap_idx"]], on="event_id", how="inner")  # only kept snapshots
    r["recipe_obs_rows_kept"] = len(m)
    r["recipe_label_null"] = int(m.x_t.isna().sum())
    pp = p[p.observed].merge(m[["snap_idx", "obj"] + [c + "_t" for c in LBL]], left_on=["snap_idx", "object_id"], right_on=["snap_idx", "obj"], how="inner")
    r["recipe_join_rows"] = len(pp)
    bad = 0
    for c in LBL:
        a = pp["lbl_" + c].to_numpy(); b = pp[c + "_t"].to_numpy().astype("float32")
        bad += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    r["recipe_label_mismatch_cells"] = bad
    # observed 360 pos in table equals objects table x,y
    mm = o.merge(s[["event_id", "snap_idx"]], on="event_id").merge(p[p.observed][["snap_idx", "object_id", "obs_x", "obs_y"]], left_on=["snap_idx", "obj"], right_on=["snap_idx", "object_id"], how="inner")
    r["obs_pos_mismatch"] = int(((mm.x.astype("float32") != mm.obs_x) | (mm.y.astype("float32") != mm.obs_y)).sum())
    # (2) brute-force timing check on a random row sample
    rng = np.random.default_rng(sbid)
    oo = o.merge(fr[["event_id", "aligned_tracking_time_s"]], on="event_id", how="left")
    oo["per"] = oo.period
    grp = {k: (g.aligned_tracking_time_s.to_numpy(), g.x.to_numpy(), g.y.to_numpy()) for k, g in oo.groupby(["obj", "per"])}
    perstr = {1: "first_half", 2: "second_half"}
    samp = p.iloc[rng.choice(len(p), 600, replace=False)]
    nbad = 0
    for row in samp.itertuples():
        t, xs, ys = grp.get((row.object_id, perstr[row.period]), (np.array([]), np.array([]), np.array([])))
        before = np.where(t < row.t_s)[0]; after = np.where(t > row.t_s)[0]
        if len(before):
            j = before[np.argmax(t[before])]; ok = row.has_prev and abs(row.prev_dt - (row.t_s - t[j])) < 1e-3 and abs(row.prev_x - xs[j]) < 1e-3 and abs(row.prev_y - ys[j]) < 1e-3
        else:
            ok = (not row.has_prev) and row.prev_dt == -1 and np.isnan(row.prev_x)
        if len(after):
            j = after[np.argmin(t[after])]; ok2 = row.has_next and abs(row.next_dt - (t[j] - row.t_s)) < 1e-3 and abs(row.next_x - xs[j]) < 1e-3 and abs(row.next_y - ys[j]) < 1e-3
        else:
            ok2 = (not row.has_next) and row.next_dt == -1 and np.isnan(row.next_x)
        nbad += (not ok) + (not ok2)
    r["timing_sample_n"] = len(samp); r["timing_sample_bad_sides"] = int(nbad)
    # (3) velocity distribution: sample of raw tracking (every 25th frame, players) vs our labels
    pl = tr[(tr.ball == False)]
    r["trk_vel_sample"] = pl[pl.frame_id % 25 == 0].vel.dropna().to_numpy().astype("float32")
    # (4) window slicing
    n = len(s); N = 5; wbad = 0
    for i in rng.choice(np.arange(N, n - N), 20, replace=False):
        lo = s.row_start.iloc[i - N]; hi = s.row_start.iloc[i + N] + s.n_rows.iloc[i + N]
        w = p.iloc[lo:hi]
        wbad += int(not (w.snap_idx.min() == i - N and w.snap_idx.max() == i + N and len(w) == s.n_rows.iloc[i - N:i + N + 1].sum()))
    r["window_bad"] = wbad
    # (5) snapshot table consistency
    r["snap_nrows_mismatch"] = int((p.groupby("snap_idx").size().to_numpy() != s.n_rows.to_numpy()).sum())
    r["nvis_eq_obsrows"] = int((s.n_visible == s.n_observed_rows).sum())
    r["period_nonmonotone"] = int((np.diff(s.period.to_numpy()) < 0).sum() + (np.diff(s.t_s.to_numpy()[s.period.to_numpy() == 1]) < 0).sum() + (np.diff(s.t_s.to_numpy()[s.period.to_numpy() == 2]) < 0).sum())
    return r


if __name__ == "__main__":
    sp = pd.read_csv(f"{OUT}/splits.csv")
    jobs = [(int(r.match_id), r.dfl_match_id) for r in sp.itertuples()]
    with Pool(8) as pool:
        res = pool.map(one, jobs, chunksize=1)
    R = pd.DataFrame(res)
    vel_trk = np.concatenate(R.pop("trk_vel_sample").to_list())
    V = {}
    # ---- aggregate from the written tables
    P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/players/*.parquet"))], ignore_index=True)
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/snapshots/*.parquet"))], ignore_index=True)
    bs = pd.read_csv(f"{OUT}/build_stats.csv")
    V["n_matches"] = int(P.match_id.nunique()); V["n_snapshots_kept"] = len(S); V["n_snapshots_total_incl_no_frame"] = int(bs.n_snapshots_total.sum())
    V["n_snapshots_dropped_no_frame"] = int(bs.n_snapshots_no_frame.sum())
    V["n_rows"] = len(P); V["22x_snapshots"] = 22 * len(S); V["rows_minus_22x"] = len(P) - 22 * len(S)
    V["rows_per_snapshot_dist"] = S.n_rows.value_counts().sort_index().to_dict()
    V["in_tracking_per_snapshot_dist"] = S.n_in_tracking.value_counts().sort_index().to_dict()
    V["rows_not_in_tracking"] = int((~P.in_tracking).sum())
    V["rows_observed"] = int(P.observed.sum()); V["rows_hidden_in_tracking"] = int((P.in_tracking & ~P.observed).sum())
    V["observed_rate_of_in_tracking_rows"] = float(P[P.in_tracking].observed.mean())
    V["mean_n_visible_players_frames"] = float(S.n_visible.mean())
    V["mean_observed_rows_per_snapshot"] = float(S.n_observed_rows.mean())
    V["mean_hidden_per_snapshot"] = float(S.n_hidden.mean())
    V["mean_in_tracking_per_snapshot"] = float(S.n_in_tracking.mean())
    # labels
    V["label_present_rate_all_rows"] = float(P.has_label.mean())
    V["label_present_rate_in_tracking"] = float(P[P.in_tracking].has_label.mean())
    V["label_present_rate_observed"] = float(P[P.observed].has_label.mean())
    V["label_present_rate_hidden_in_tracking"] = float(P[P.in_tracking & ~P.observed].has_label.mean())
    V["label_nan_x_given_in_tracking"] = int((P.in_tracking & P.lbl_x.isna()).sum())
    V["label_nan_vx_given_has_label"] = int((P.has_label & (P.lbl_vx.isna() | P.lbl_ax.isna())).sum())
    q = [0.5, 0.9, 0.99, 0.999]
    V["vel_tracking_sample_quantiles"] = dict(zip(map(str, q), np.quantile(vel_trk, q).astype(float))) | {"mean": float(vel_trk.mean()), "max": float(vel_trk.max()), "n": len(vel_trk)}
    lv = P.lbl_vel.dropna()
    V["vel_table_quantiles"] = dict(zip(map(str, q), np.quantile(lv, q).astype(float))) | {"mean": float(lv.mean()), "max": float(lv.max()), "n": len(lv)}
    for k, msk in {"observed": P.observed, "hidden": P.in_tracking & ~P.observed}.items():
        v = P[msk].lbl_vel.dropna(); V[f"vel_{k}_mean_median"] = [float(v.mean()), float(v.median())]
    V["max_abs_vx_vy"] = float(max(P.lbl_vx.abs().max(), P.lbl_vy.abs().max())); V["max_abs_ax_ay"] = float(max(P.lbl_ax.abs().max(), P.lbl_ay.abs().max()))
    # 360 position error vs label
    ob_ = P[P.observed & P.has_label]
    e = np.hypot(ob_.obs_x - ob_.lbl_x, ob_.obs_y - ob_.lbl_y)
    V["pos_err_pooled_median_m"] = float(e.median()); V["pos_err_excl_weak_median_m"] = float(e[~ob_.align_weak].median())
    V["pos_err_rms_m"] = float(np.sqrt((e ** 2).mean())); V["pos_err_p99_m"] = float(e.quantile(0.99))
    pm = e.groupby(ob_.match_id).median(); V["pos_err_per_match_median_range_m"] = [float(pm.min()), float(pm.max())]
    V["pos_err_weak_matches_median_of_medians_m"] = float(pm[pm.index.isin(sp[sp.align_weak].match_id)].median())
    V["pos_err_nonweak_median_of_medians_m"] = float(pm[~pm.index.isin(sp[sp.align_weak].match_id)].median())
    # timing
    ob2 = P[P.observed]
    g = ob2.prev_dt[ob2.has_prev]; V["obs_prev_dt_quantiles"] = {k: float(g.quantile(k)) for k in (0.5, 0.85, 0.99, 0.999)}
    g = ob2.next_dt[ob2.has_next]; V["obs_next_dt_quantiles"] = {k: float(g.quantile(k)) for k in (0.5, 0.85, 0.99, 0.999)}
    both = ob2.has_prev & ob2.has_next
    V["obs_both_sides_rate"] = float(both.mean()); V["obs_one_side_only_rate"] = float((ob2.has_prev ^ ob2.has_next).mean()); V["obs_no_side_rate"] = float((~ob2.has_prev & ~ob2.has_next).mean())
    V["obs_both_within_5s"] = float((both & (ob2.prev_dt <= 5) & (ob2.next_dt <= 5)).mean()); V["obs_both_within_2s"] = float((both & (ob2.prev_dt <= 2) & (ob2.next_dt <= 2)).mean())
    hd = P[P.in_tracking & ~P.observed]
    V["hidden_rows_both_sides_rate"] = float((hd.has_prev & hd.has_next).mean()); V["hidden_one_side_rate"] = float((hd.has_prev ^ hd.has_next).mean()); V["hidden_no_side_rate"] = float((~hd.has_prev & ~hd.has_next).mean())
    V["hidden_prev_dt_quantiles"] = {k: float(hd.prev_dt[hd.has_prev].quantile(k)) for k in (0.5, 0.85, 0.99)}
    V["sentinel_violations"] = int(((~P.has_prev) & (P.prev_dt != -1)).sum() + ((P.has_prev) & (P.prev_dt < 0)).sum() + ((~P.has_next) & (P.next_dt != -1)).sum() + ((P.has_next) & (P.next_dt < 0)).sum())
    V["obs_nan_when_observed"] = int((P.observed & P.obs_x.isna()).sum()); V["obs_present_when_hidden"] = int((~P.observed & P.obs_x.notna()).sum())
    # flags
    V["observed_not_in_tracking_rows"] = int((P.observed & ~P.in_tracking).sum())
    V["lineup_pos_missing_rate_in_tracking"] = float(P[P.in_tracking].pos_id.isna().mean())
    V["tracking_rows_with_lineup_pitch_disagree"] = {"lineup_pitch_not_tracking": int(bs.n_lineup_pitch_not_tracking.sum()), "tracking_not_lineup_pitch": int(bs.n_tracking_not_lineup_pitch.sum())}
    V["agree_360_flags_min"] = {c: float(bs[c].min()) for c in ("agree_teammate", "agree_actor", "agree_keeper")}
    V["n_hidden_but_observed_same_t"] = int(bs.n_hidden_but_observed_same_t.sum())
    V["same_t_pos_maxdiff_m_max"] = float(bs.obs_sametime_pos_maxdiff_m.max())
    V["teammate_rows_per_snapshot_mean"] = float(P.groupby(["match_id", "snap_idx"]).teammate.sum().mean())
    V["keepers_per_snapshot_dist"] = P.groupby(["match_id", "snap_idx"]).keeper.sum().value_counts().sort_index().to_dict()
    # split / leakage
    sp_ids = {k: set(v.match_id) for k, v in sp.groupby("split")}
    V["split_matches"] = {k: len(v) for k, v in sp_ids.items()}
    V["split_disjoint"] = bool(not (sp_ids["train"] & sp_ids["valid"] or sp_ids["train"] & sp_ids["test"] or sp_ids["valid"] & sp_ids["test"]))
    V["split_union_is_330_files"] = bool((sp_ids["train"] | sp_ids["valid"] | sp_ids["test"]) == set(P.match_id.unique()) and len(P.match_id.unique()) == 330)
    V["row_split_matches_splits_csv"] = bool((P.drop_duplicates("match_id").set_index("match_id").split == sp.set_index("match_id").split).all())
    V["split_rows"] = P.groupby("split").size().to_dict(); V["split_snapshots"] = S.groupby("split").size().to_dict()
    V["split_weak_matches"] = sp[sp.align_weak].groupby("split").size().to_dict()
    V["event_id_overlap_across_splits"] = int(S.event_id.duplicated().sum())
    V["excluded_ids_present"] = sorted(set(P.match_id.unique()) & {4077658, 4077659, 4077660, 4077677, 4077679})
    V["weak_flag_ids"] = sorted(P[P.align_weak].match_id.unique().tolist())
    # per-match checks
    V["recipe_rows_kept_total"] = int(R.recipe_obs_rows_kept.sum()); V["recipe_label_null_total"] = int(R.recipe_label_null.sum())
    V["recipe_join_rows_total"] = int(R.recipe_join_rows.sum()); V["recipe_label_mismatch_cells_total"] = int(R.recipe_label_mismatch_cells.sum())
    V["obs_pos_mismatch_total"] = int(R.obs_pos_mismatch.sum())
    V["timing_sample_rows"] = int(R.timing_sample_n.sum()); V["timing_sample_bad_sides"] = int(R.timing_sample_bad_sides.sum())
    V["window_bad_total"] = int(R.window_bad.sum()); V["snap_nrows_mismatch_total"] = int(R.snap_nrows_mismatch.sum())
    V["snapshots_where_nvisible_eq_observed_rows"] = [int(R.nvis_eq_obsrows.sum()), len(S)]
    V["time_order_violations"] = int(R.period_nonmonotone.sum())
    # attack-direction normalisation: event-team goalkeeper must sit at negative x, opponent goalkeeper at positive x, in both periods
    V["atk_flip_fraction_snapshots"] = float(S.atk_flip.mean()); V["atk_flip_snapshots_true"] = int(S.atk_flip.sum())
    V["atk_flip_equals_not_actor_is_home"] = bool((S.atk_flip == ~S.actor_is_home).all())
    tmr = P[P.teammate]; V["atk_flip_rows_vs_event_team_is_home_violations"] = int((tmr.atk_flip == tmr.is_home).sum())
    V["atk_flip_players_vs_snapshots_violations"] = int((P.groupby(["match_id", "snap_idx"]).atk_flip.first().to_numpy() != S.set_index(["match_id", "snap_idx"]).loc[P.groupby(["match_id", "snap_idx"]).atk_flip.first().index].atk_flip.to_numpy()).sum())
    gkp = P[P.keeper & P.has_label]
    V["gk_lbl_x_median_event_team_vs_opponent_by_period"] = {f"period{per}_{'event_team' if tm else 'opponent'}": float(h.lbl_x.median()) for (per, tm), h in gkp.groupby(["period", "teammate"])}
    # storage
    pb = sum(os.path.getsize(f) for f in glob.glob(f"{OUT}/players/*.parquet")); sb = sum(os.path.getsize(f) for f in glob.glob(f"{OUT}/snapshots/*.parquet"))
    V["bytes_players"] = pb; V["bytes_snapshots"] = sb
    V["players_in_memory_bytes"] = int(P.memory_usage(deep=True).sum())
    for N in (5, 10, 20): V[f"materialised_windows_N{N}_compressed_bytes_est"] = pb * (2 * N + 1)
    json.dump(V, open(f"{OUT}/verification.json", "w"), indent=1, default=str)
    for k, v in V.items(): print(k, v)
