"""Build per-(snapshot, player) tables for one or many matches.
Usage: build_dataset.py [--workers N] [sbid ...]   (no ids = all 56 in splits.csv)"""
import glob, json, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ROOT = "/hpcwork/p0029161/scoutgpt/_cdfbatch/out/2026-2027"
LINEUPS = "/hpcwork/p0029161/statsbomb/bundesliga/raw/lineups"
OUT = "/hpcwork/p0029161/scoutgpt/_phase2_dataset/out"
LBL = ["x", "y", "vx", "vy", "ax", "ay", "vel", "acc"]


def pos_group(pid):
    if pid == 1: return "GK"
    if 2 <= pid <= 8: return "DEF"
    if 9 <= pid <= 16 or 18 <= pid <= 20: return "MID"
    if pid == 17 or 21 <= pid <= 25: return "FWD"
    return None


def ts_secs(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def lineup_segments(sbid):
    """(sb_player_id, position_id, K_from, K_to) with K = period*1e4 + period-relative seconds."""
    f = glob.glob(f"{LINEUPS}/{sbid}_*.json")
    assert len(f) >= 1, f
    f = [max(f, key=lambda p: int(p.rsplit("_", 1)[1][:-5]))]   # newest version token, as in the manifest
    segs = []
    for team in json.load(open(f[0])):
        for p in team["lineup"]:
            for q in p["positions"]:
                assert q["from_period"] in (1, 2) and q["to_period"] in (None, 1, 2)
                kf = q["from_period"] * 1e4 + ts_secs(q["from"]) - (2700 if q["from_period"] == 2 else 0)
                kt = np.inf if q["to"] is None else q["to_period"] * 1e4 + ts_secs(q["to"]) - (2700 if q["to_period"] == 2 else 0)
                segs.append((p["player_id"], q["position_id"], kf, kt))
    return segs


def build(args):
    sbid, dfl, split, weak = args
    t0 = time.time()
    d = f"{ROOT}/{sbid}_{dfl}"
    fr = pd.read_parquet(f"{d}/sb360_frames.parquet")
    ob = pd.read_parquet(f"{d}/sb360_objects.parquet")
    ev = pd.read_parquet(f"{d}/sb_events.parquet", columns=["event_id", "type", "sub_type", "outcome"])
    lj = pd.read_parquet(f"{d}/lineup_join.parquet")
    tr = pd.read_parquet(f"{d}/tracking.parquet", columns=["frame_id", "ball", "object_id", "x", "y", "z", "vx", "vy", "vel", "ax", "ay", "acc", "ball_status"])
    st = {"match_id": sbid, "dfl_match_id": dfl, "split": split, "align_weak": weak}

    # ---------- tracking: players and ball ----------
    ball = tr[tr.ball][["frame_id", "x", "y", "z", "ball_status"]].rename(columns={"x": "lbl_ball_x", "y": "lbl_ball_y", "z": "lbl_ball_z", "ball_status": "lbl_ball_status"})
    assert ball.frame_id.is_unique
    pl = tr[~tr.ball].drop(columns=["ball", "z", "ball_status"]).copy()
    pl["object_id"] = pl.object_id.astype(str)
    pl = pl.rename(columns={c: "lbl_" + c for c in LBL})
    frames_in_tracking = set(pl.frame_id.unique())

    # ---------- identity ----------
    lj = lj.copy()
    sb2obj = dict(zip(lj.sb_player_id.astype(str), lj.dfl_object_id.astype(str)))
    ident = pd.DataFrame({"object_id": lj.dfl_object_id.astype(str), "sb_player_id": lj.sb_player_id.astype("int64"),
                          "team_id": lj.sb_team_id.astype("int32"), "is_home": lj.home_away.eq("home").to_numpy(),
                          "jersey": lj.uniform_number.astype("int16")})
    assert ident.object_id.is_unique

    # ---------- snapshots (all 360 frames) ----------
    fr = fr.copy()
    fr["period_i"] = fr.period.map({"first_half": 1, "second_half": 2})
    assert fr.period_i.notna().all()
    fr["period_i"] = fr.period_i.astype("int8")
    fr["t_s"] = fr.aligned_tracking_time_s.astype("float64")
    fr["frame_id"] = fr.aligned_tracking_frame_id.astype("float64")
    fr = fr.sort_values(["period_i", "t_s", "event_index"], kind="stable").reset_index(drop=True)
    st["n_snapshots_total"] = len(fr)
    keep = fr.frame_id.notna() & fr.frame_id.isin(frames_in_tracking) & fr.t_s.notna()
    st["n_snapshots_no_frame"] = int((~keep).sum())
    fr["snap_idx"] = -1
    fr.loc[keep, "snap_idx"] = np.arange(int(keep.sum()))
    fr["frame_id"] = fr.frame_id.astype("Int64")
    # ---------- observations ----------
    o = ob.merge(fr[["event_id", "snap_idx", "period_i", "t_s", "event_index"]], on="event_id", how="left", suffixes=("_ob", ""))
    assert o.t_s.notna().all()
    o["object_id"] = o.player_id.astype(str).map(sb2obj)
    st["n_obs_rows"] = len(o)
    st["n_obs_unmapped"] = int(o.object_id.isna().sum())
    assert st["n_obs_unmapped"] == 0
    # unique observations per (object, period, time) for timing lookup
    ou = o.sort_values(["event_index"]).groupby(["object_id", "period_i", "t_s"], as_index=False, sort=True).agg(
        x=("x", "first"), y=("y", "first"), xmax=("x", "max"), xmin=("x", "min"), ymax=("y", "max"), ymin=("y", "min"))
    st["obs_sametime_pos_maxdiff_m"] = float(max((ou.xmax - ou.xmin).max(), (ou.ymax - ou.ymin).max()))
    obs_groups = {k: g for k, g in ou.sort_values("t_s").groupby(["object_id", "period_i"])}

    # ---------- row universe: (tracking at aligned frame) U (observed) ----------
    sk = fr[keep][["snap_idx", "frame_id"]].copy()
    sk["frame_id"] = sk.frame_id.astype("int64")
    trk = sk.merge(pl.assign(frame_id=pl.frame_id.astype("int64")), on="frame_id", how="left")
    trk["in_tracking"] = True
    ok = o[o.snap_idx >= 0][["snap_idx", "object_id", "x", "y", "in_pitch"]].rename(columns={"x": "obs_x", "y": "obs_y", "in_pitch": "obs_in_pitch"})
    ok["observed"] = True
    rows = trk.merge(ok, on=["snap_idx", "object_id"], how="outer")
    rows["in_tracking"] = rows.in_tracking.fillna(False).astype(bool)
    rows["observed"] = rows.observed.fillna(False).astype(bool)
    rows = rows.drop(columns="frame_id")
    st["n_rows"] = len(rows)

    # ---------- snapshot-level attributes onto rows ----------
    sn = fr[keep].copy()
    rows = rows.merge(sn[["snap_idx", "period_i", "t_s", "actor_team_id", "actor_player_id", "match_clock_seconds"]], on="snap_idx", how="left")
    rows = rows.merge(ident, on="object_id", how="left")
    assert rows.sb_player_id.notna().all()
    rows["teammate"] = rows.team_id.astype("int64").astype(str) == rows.actor_team_id.astype(str)
    rows["actor"] = rows.sb_player_id.astype(str) == rows.actor_player_id.astype(str)

    # ---------- lineup position at the snapshot time ----------
    segs = lineup_segments(sbid)
    Ksnap = (sn.period_i.astype(float) * 1e4 + sn.match_clock_seconds).to_numpy()
    parts = []
    for pid, posid, kf, kt in segs:
        m = (Ksnap >= kf) & (Ksnap < kt)
        if m.any():
            parts.append(pd.DataFrame({"snap_idx": sn.snap_idx.to_numpy()[m], "sb_player_id": np.int64(pid), "pos_id": np.int16(posid)}))
    pos = pd.concat(parts, ignore_index=True)
    st["lineup_dup_pos"] = int(pos.duplicated(["snap_idx", "sb_player_id"]).sum())
    pos = pos.drop_duplicates(["snap_idx", "sb_player_id"])
    rows = rows.merge(pos, on=["snap_idx", "sb_player_id"], how="left")
    rows["pos_id"] = rows.pos_id.astype("Int16")
    rows["pos_group"] = rows.pos_id.map(lambda v: None if pd.isna(v) else pos_group(int(v)))
    rows["keeper"] = rows.pos_group.eq("GK")
    # lineup says on pitch, tracking says not (and vice versa) -> reported, not filtered
    st["n_lineup_pitch_not_tracking"] = int((rows.pos_id.notna() & ~rows.in_tracking).sum())
    st["n_tracking_not_lineup_pitch"] = int((rows.pos_id.isna() & rows.in_tracking).sum())
    # 360 flags for agreement check (observed rows only)
    chk = o[o.snap_idx >= 0][["snap_idx", "object_id", "teammate", "actor", "keeper"]].rename(columns=lambda c: c if c in ("snap_idx", "object_id") else c + "_360")
    cm = rows.merge(chk, on=["snap_idx", "object_id"], how="inner")
    st["agree_teammate"] = float((cm.teammate == cm.teammate_360).mean())
    st["agree_actor"] = float((cm.actor == cm.actor_360).mean())
    st["agree_keeper"] = float((cm.keeper == cm.keeper_360).mean())

    # ---------- timing features ----------
    n = len(rows)
    prev_dt = np.full(n, -1.0, "float64"); next_dt = np.full(n, -1.0, "float64")
    prev_x = np.full(n, np.nan); prev_y = np.full(n, np.nan); next_x = np.full(n, np.nan); next_y = np.full(n, np.nan)
    has_prev = np.zeros(n, bool); has_next = np.zeros(n, bool)
    tt_all = rows.t_s.to_numpy()
    for (obj, per), idx in rows.groupby(["object_id", "period_i"]).indices.items():
        g = obs_groups.get((obj, per))
        if g is None:
            continue
        tms = g.t_s.to_numpy(); gx = g.x.to_numpy(); gy = g.y.to_numpy()
        tt = tt_all[idx]
        ip = np.searchsorted(tms, tt, "left") - 1          # last observation strictly before t
        inx = np.searchsorted(tms, tt, "right")            # first observation strictly after t
        hp = ip >= 0; hn = inx < len(tms)
        i1 = idx[hp]; has_prev[i1] = True; prev_dt[i1] = tt[hp] - tms[ip[hp]]; prev_x[i1] = gx[ip[hp]]; prev_y[i1] = gy[ip[hp]]
        i2 = idx[hn]; has_next[i2] = True; next_dt[i2] = tms[inx[hn]] - tt[hn]; next_x[i2] = gx[inx[hn]]; next_y[i2] = gy[inx[hn]]
    rows["has_prev"] = has_prev; rows["prev_dt"] = prev_dt; rows["prev_x"] = prev_x; rows["prev_y"] = prev_y
    rows["has_next"] = has_next; rows["next_dt"] = next_dt; rows["next_x"] = next_x; rows["next_y"] = next_y
    # hidden here but observed at the identical time in another snapshot (not used by strict prev/next)
    key_obs = set(zip(ou.object_id, ou.period_i, ou.t_s))
    hid = rows[~rows.observed]
    st["n_hidden_but_observed_same_t"] = int(sum((a, b, c) in key_obs for a, b, c in zip(hid.object_id, hid.period_i, hid.t_s)))

    # ---------- labels ----------
    rows["has_label"] = rows.in_tracking & rows.lbl_x.notna() & rows.lbl_y.notna()
    rows["lbl_heading"] = np.arctan2(rows.lbl_vy, rows.lbl_vx)
    rows["match_id"] = np.int32(sbid)
    rows["period"] = rows.period_i
    rows["align_weak"] = weak
    rows["split"] = split
    rows = rows.sort_values(["snap_idx", "object_id"], kind="stable").reset_index(drop=True)

    # ---------- attack-direction normalisation ----------
    # The tables come out of tracking in the static_home_left frame (home attacks +x in both halves). Event team = actor team.
    # When the event team is the away team the WHOLE snapshot is rotated 180 deg about the pitch centre, so the event team attacks +x.
    # One flip per snapshot, applied uniformly to every column of every row of that snapshot: prev_*/next_* come from other
    # snapshots (possibly with the other possession) but are expressed in THIS snapshot's frame, never in their own.
    ev_home = sn.set_index("snap_idx").actor_team_id.astype(str).map(dict(zip(ident.team_id.astype(str), ident.is_home)))
    st["n_snap_team_unassigned"] = int(ev_home.isna().sum())            # no event team -> atk_flip False, counted here
    tmh = rows[rows.teammate].groupby("snap_idx").is_home.agg(["min", "max"])   # cross-check from the players table itself
    st["n_snap_team_conflict"] = int(((tmh["min"] != tmh["max"]) | (tmh["min"] != ev_home.reindex(tmh.index).astype(object))).sum())
    flip_snap = ev_home.eq(False)
    rows["atk_flip"] = rows.snap_idx.map(flip_snap).astype(bool)

    cols = ["match_id", "snap_idx", "period", "t_s", "object_id", "sb_player_id", "team_id", "is_home", "jersey", "pos_id", "pos_group",
            "teammate", "actor", "keeper", "in_tracking", "observed", "obs_x", "obs_y", "obs_in_pitch",
            "has_prev", "prev_dt", "prev_x", "prev_y", "has_next", "next_dt", "next_x", "next_y",
            "has_label", "lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay", "lbl_vel", "lbl_acc", "lbl_heading", "align_weak", "split", "atk_flip"]
    out = rows[cols].copy()
    fl = out.atk_flip.to_numpy()
    for c in ["obs_x", "obs_y", "prev_x", "prev_y", "next_x", "next_y", "lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay"]:
        out[c] = np.where(fl, -out[c], out[c])
    h = out.lbl_heading.to_numpy()                                       # shift by pi, wrapped into (-pi, pi]
    out["lbl_heading"] = np.where(fl, np.where(h > 0, h - np.pi, h + np.pi), h)
    f32 = ["obs_x", "obs_y", "prev_dt", "prev_x", "prev_y", "next_dt", "next_x", "next_y"] + [c for c in out.columns if c.startswith("lbl_")]
    out[f32] = out[f32].astype("float32")
    out["obs_in_pitch"] = out.obs_in_pitch.astype("boolean")
    out["match_id"] = out.match_id.astype("int32")

    # ---------- snapshots table ----------
    g = rows.groupby("snap_idx")
    cnt = pd.DataFrame({"n_rows": g.size(), "n_in_tracking": g["in_tracking"].sum(), "n_observed_rows": g["observed"].sum(),
                        "n_hidden": g.apply(lambda x: int((x.in_tracking & ~x.observed).sum()), include_groups=False)}).reset_index()
    cnt["row_start"] = cnt.n_rows.cumsum() - cnt.n_rows
    ar = rows[rows.actor & rows.observed][["snap_idx", "obs_x", "obs_y"]].drop_duplicates("snap_idx").rename(columns={"obs_x": "actor_x", "obs_y": "actor_y"})
    s = sn.drop(columns="period").merge(ev, on="event_id", how="left").merge(cnt, on="snap_idx", how="left").merge(ar, on="snap_idx", how="left")
    bl = ball.assign(frame_id=ball.frame_id.astype("int64"))
    s["frame_id"] = s.frame_id.astype("int64")
    s = s.merge(bl, on="frame_id", how="left")
    s["match_id"] = np.int32(sbid)
    s["actor_team_id"] = s.actor_team_id.astype("int32")
    s["actor_player_id"] = s.actor_player_id.astype("int64")
    s["actor_is_home"] = s.actor_team_id.map(dict(zip(ident.team_id, ident.is_home)))
    s = s.rename(columns={"period_i": "period", "type": "event_type", "sub_type": "event_sub_type", "outcome": "event_outcome", "n_visible_players": "n_visible"})
    s["event_index"] = s.event_index.astype("int32")
    s["align_weak"] = weak; s["split"] = split
    scols = ["match_id", "snap_idx", "event_id", "event_index", "period", "match_clock_seconds", "t_s", "frame_id", "offset_s",
             "event_type", "event_sub_type", "event_outcome", "actor_team_id", "actor_player_id", "actor_is_home", "actor_x", "actor_y",
             "n_visible", "n_rows", "n_in_tracking", "n_observed_rows", "n_hidden", "row_start",
             "lbl_ball_x", "lbl_ball_y", "lbl_ball_z", "lbl_ball_status", "align_weak", "split", "atk_flip"]
    s["atk_flip"] = s.snap_idx.map(flip_snap).astype(bool)
    s = s[scols].sort_values("snap_idx").reset_index(drop=True)
    for c in ["event_type", "event_sub_type", "event_outcome"]:
        s[c] = s[c].astype("object").where(s[c].notna(), None)
    for c in ["match_clock_seconds", "offset_s", "actor_x", "actor_y", "lbl_ball_x", "lbl_ball_y", "lbl_ball_z"]:
        s[c] = s[c].astype("float32")
    s["n_visible"] = s.n_visible.astype("int16")
    for c in ["actor_x", "actor_y", "lbl_ball_x", "lbl_ball_y"]:   # lbl_ball_z is height: invariant
        s[c] = np.where(s.atk_flip, -s[c], s[c]).astype("float32")
    assert (s.row_start.to_numpy()[1:] == (s.row_start + s.n_rows).to_numpy()[:-1]).all()
    assert s.n_rows.sum() == len(out)
    assert out.snap_idx.is_monotonic_increasing

    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), f"{OUT}/players/{sbid}.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(s, preserve_index=False), f"{OUT}/snapshots/{sbid}.parquet", compression="zstd")
    st["n_snapshots"] = len(s)
    st["n_observed"] = int(out.observed.sum())
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st


if __name__ == "__main__":
    a = sys.argv[1:]
    w = 1
    if a and a[0] == "--workers":
        w = int(a[1]); a = a[2:]
    sp = pd.read_csv(f"{OUT}/splits.csv")
    if a:
        sp = sp[sp.match_id.isin([int(x) for x in a])]
    jobs = [(int(r.match_id), r.dfl_match_id, r.split, bool(r.align_weak)) for r in sp.itertuples()]
    if w > 1:
        with Pool(w) as p:
            res = p.map(build, jobs, chunksize=1)
    else:
        res = [build(j) for j in jobs]
    df = pd.DataFrame(res)
    name = "build_stats.csv" if len(jobs) == 56 else "build_stats_partial.csv"
    df.to_csv(f"{OUT}/{name}", index=False)
    print(df.drop(columns=["dfl_match_id"]).to_string())
