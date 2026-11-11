"""Phase-2 dataset -> PlayerImputer SetTransformer loader.

Source: /hpcwork/p0029161/scoutgpt/_phase2_dataset/out/players/<sb_match_id>.parquet (contract: SCHEMA.md there, read-only).
One sample = W consecutive 360 snapshots of one match (same period) around a target snapshot; the loss uses the middle one.

Sample / batch keys (see collate_phase2 for batches; shapes without the leading B for a sample):
    features               (W, N, F) float32   FEATURES, last column = pos_id (categorical index, 0 = unknown / absent)
    obs_xy                 (W, N, 2) float32   360 (x/105, y/68), pitch-centre origin, attack-normalised; 0 where not visible
    visible                (W, N)    bool      player is in this snapshot's freeze frame (`observed`)
    valid                  (W, N)    bool      player has a row in this snapshot (padding / absent player / absent timestep: False)
    labels                 (W, N, 6) float32   (vx, vy, x/105, y/68, ax, ay); NaN where ~has_label or ~valid
    actor_global_index_lst ()        int64     slot of the event actor in the sample (-1 if none); ignored by the loss
    (not collated)  categorical_indices, categorical_features, object_ids, snap_idx_window (W,) int64 with -1 for absent timesteps

LEAKAGE RULE (decided by the project owner; do not relax):
  * Features come only from the `I`-marked players columns. prev_*/next_* are 360 *sightings* of the same player in other snapshots of the
    same match and period. They come from StatsBomb 360, not from tracking, so they exist in leagues without any tracking. They are the core
    signal (Phase 0: R^2 0.57-0.63 from central differencing them), not leakage.
  * Never a feature: every `lbl_*`; `in_tracking` and `has_label` (they encode "DFL tracking has this player", which does not exist outside
    the Bundesliga; target filter only); `align_weak`, `split`, `match_id`, `dfl_match_id` (bookkeeping).
  * `sb_player_id` is never a learned per-player embedding (it cannot transfer to other leagues and overfits 56 matches). Identity is used only
    for association across snapshots (`object_id`), which is done here. Role comes from pos_id, is_home, teammate, keeper, actor.
    `team_id` and `jersey` are identity-like as well and are left out.
"""
import os
import bisect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import imputer.config as config

PHASE2_ROOT = "/hpcwork/p0029161/scoutgpt/_phase2_dataset"

# feature name -> players columns it is computed from (explicit so that a test can assert against FORBIDDEN_COLUMNS)
FEATURE_SOURCES = {
    "is_home": ["is_home"],
    "teammate": ["teammate"],
    "actor": ["actor"],
    "keeper": ["keeper"],
    "has_prev": ["has_prev"],
    "has_next": ["has_next"],
    "prev_dt_log": ["prev_dt", "has_prev"],  # standardised log1p(prev_dt); 0 where ~has_prev (the -1.0 sentinel never enters)
    "next_dt_log": ["next_dt", "has_next"],
    "prev_x": ["prev_x", "has_prev"],  # x/105, 0 where ~has_prev
    "prev_y": ["prev_y", "has_prev"],
    "next_x": ["next_x", "has_next"],
    "next_y": ["next_y", "has_next"],
    "dt_rel": ["t_s", "period"],  # signed log1p of the seconds between this snapshot and the target snapshot, / RMS (train); 0 if absent
    "present": [],  # 1 if the player has a row in this snapshot (same as `valid`); lets the time attention recognise neutral slots
    "pos_id": ["pos_id"],  # categorical, null -> 0
}
FEATURES = list(FEATURE_SOURCES)
F_DIM = len(FEATURES)
I_PREV_DT, I_NEXT_DT, I_DT_REL, I_PRESENT, I_POS = (FEATURES.index(n) for n in ("prev_dt_log", "next_dt_log", "dt_rel", "present", "pos_id"))
CATEGORICAL_INDICES = [I_POS]
CATEGORICAL_FEATURES = {"pos_id": 25}  # StatsBomb position ids 1..24 (25 allowed), 0 = unknown / absent

OBS_SOURCES = ["obs_x", "obs_y", "observed"]
LABEL_COLUMNS = ["lbl_vx", "lbl_vy", "lbl_x", "lbl_y", "lbl_ax", "lbl_ay"]  # order of the model output
LABEL_SOURCES = LABEL_COLUMNS + ["has_label"]  # has_label is the target filter
KEY_COLUMNS = ["snap_idx", "object_id", "period", "t_s"]
FORBIDDEN_COLUMNS = frozenset(
    ["lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay", "lbl_vel", "lbl_acc", "lbl_heading",
     "in_tracking", "has_label", "align_weak", "split", "match_id", "dfl_match_id", "sb_player_id", "team_id", "jersey"]
)
READ_COLUMNS = sorted(set(KEY_COLUMNS + OBS_SOURCES + LABEL_SOURCES + [c for v in FEATURE_SOURCES.values() for c in v]))


def read_splits(root=PHASE2_ROOT):
    """splits.csv is the source of truth: {match_id: split}."""
    df = pd.read_csv(os.path.join(root, "out", "splits.csv"))
    return dict(zip(df["match_id"].astype(int), df["split"]))


def read_match(match_id, root=PHASE2_ROOT):
    return pd.read_parquet(os.path.join(root, "out", "players", f"{match_id}.parquet"), columns=READ_COLUMNS)


def _snapshot_axes(df):
    """snapshot ids (sorted), per-snapshot period and t_s, and each row's snapshot position."""
    snap_ids, snap_pos = np.unique(df["snap_idx"].values, return_inverse=True)
    period = np.zeros(len(snap_ids), dtype=np.int64)
    t_s = np.zeros(len(snap_ids), dtype=np.float64)
    period[snap_pos] = df["period"].values
    t_s[snap_pos] = df["t_s"].values
    return snap_ids, snap_pos, period, t_s


def _signed_log(d):
    return np.sign(d) * np.log1p(np.abs(d))


def fit_stats(match_ids, window, root=PHASE2_ROOT):
    """Normalisation statistics from TRAIN matches only (raises if a match is not a train match in splits.csv)."""
    splits = read_splits(root)
    not_train = [m for m in match_ids if splits.get(int(m)) != "train"]
    if not_train:
        raise ValueError(f"fit_stats got non-train matches: {not_train}")
    h = window // 2
    log_prev, log_next, rel = [], [], []
    for m in match_ids:
        df = read_match(m, root)
        hp, hn = df["has_prev"].values, df["has_next"].values
        log_prev.append(np.log1p(df["prev_dt"].values[hp].astype(np.float64)))
        log_next.append(np.log1p(df["next_dt"].values[hn].astype(np.float64)))
        _, _, period, t_s = _snapshot_axes(df)
        for k in range(1, h + 1):
            same = period[k:] == period[:-k]
            d = t_s[k:] - t_s[:-k]
            rel.append(_signed_log(d[same]))
    log_prev, log_next, rel = np.concatenate(log_prev), np.concatenate(log_next), np.concatenate(rel)
    return {
        "window": int(window),
        "fit_matches": [int(m) for m in match_ids],
        "log_prev_dt": {"mean": float(log_prev.mean()), "std": float(log_prev.std()), "n": int(len(log_prev))},
        "log_next_dt": {"mean": float(log_next.mean()), "std": float(log_next.std()), "n": int(len(log_next))},
        "dt_rel_rms": float(np.sqrt(np.mean(np.concatenate([rel, -rel]) ** 2))),  # offsets are symmetric (+-k), so RMS = std about 0
    }


class MatchArrays:
    """Dense (snapshot, player-slot) arrays of one match. Slot = object_id of the whole match (sorted), so players are aligned across snapshots."""

    def __init__(self, match_id, df, stats):
        L, Wd = config.field_length, config.field_width
        snap_ids, sp, period, t_s = _snapshot_axes(df)
        self.match_id, self.snap_ids, self.period, self.t_s = int(match_id), snap_ids, period, t_s
        self.object_ids, sl = np.unique(df["object_id"].values, return_inverse=True)
        S, P, n = len(snap_ids), len(self.object_ids), len(df)
        if len(np.unique(sp * P + sl)) != n:
            raise ValueError(f"match {match_id}: duplicate (snap_idx, object_id) rows")
        self.S, self.P = S, P

        hp, hn = df["has_prev"].values, df["has_next"].values
        f = np.zeros((n, F_DIM), dtype=np.float32)
        for name in ("is_home", "teammate", "actor", "keeper", "has_prev", "has_next"):
            f[:, FEATURES.index(name)] = df[name].values
        # -1.0 sentinel: never enters log1p (replaced by 0 before the transform and zeroed afterwards)
        for col, has, i, key in (("prev_dt", hp, I_PREV_DT, "log_prev_dt"), ("next_dt", hn, I_NEXT_DT, "log_next_dt")):
            z = (np.log1p(np.where(has, df[col].values, 0.0)) - stats[key]["mean"]) / stats[key]["std"]
            f[:, i] = np.where(has, z, 0.0)
        for name, has, col, scale in (("prev_x", hp, "prev_x", L), ("prev_y", hp, "prev_y", Wd), ("next_x", hn, "next_x", L), ("next_y", hn, "next_y", Wd)):
            f[:, FEATURES.index(name)] = np.where(has, df[col].values / scale, 0.0)
        f[:, I_PRESENT] = 1.0
        pos = df["pos_id"].fillna(0).values.astype(np.float32)
        if pos.max() > CATEGORICAL_FEATURES["pos_id"]:
            raise ValueError(f"match {match_id}: pos_id {pos.max()} exceeds {CATEGORICAL_FEATURES['pos_id']}")
        f[:, I_POS] = pos
        if not np.isfinite(f).all():
            raise ValueError(f"match {match_id}: non-finite feature")

        vis = df["observed"].values
        obs = np.stack([np.where(vis, df["obs_x"].values / L, 0.0), np.where(vis, df["obs_y"].values / Wd, 0.0)], axis=-1).astype(np.float32)
        lab = np.stack([df["lbl_vx"].values, df["lbl_vy"].values, df["lbl_x"].values / L, df["lbl_y"].values / Wd,
                        df["lbl_ax"].values, df["lbl_ay"].values], axis=-1).astype(np.float32)
        lab[~df["has_label"].values] = np.nan  # target filter

        self.feats = np.zeros((S, P, F_DIM), dtype=np.float32)
        self.obs = np.zeros((S, P, 2), dtype=np.float32)
        self.vis = np.zeros((S, P), dtype=bool)
        self.valid = np.zeros((S, P), dtype=bool)
        self.actor = np.zeros((S, P), dtype=bool)
        self.labels = np.full((S, P, 6), np.nan, dtype=np.float32)
        self.feats[sp, sl], self.obs[sp, sl], self.vis[sp, sl], self.valid[sp, sl] = f, obs, vis, True
        self.actor[sp, sl] = df["actor"].values
        self.labels[sp, sl] = lab


class Phase2Windows(Dataset):
    """Windows of `window` (odd) consecutive snapshots of one match and period; every snapshot is a target once.

    Edges: the target is always at index window//2. Timesteps that fall before the first / after the last snapshot of the match, or in the
    other period (t_s restarts each half), are filled with an all-invalid, all-neutral timestep (nothing is shifted or clipped).
    Neutral fill for a slot that is not valid at a timestep: features 0 (pos_id 0 = padding embedding, present 0), obs_xy 0, visible False,
    valid False, labels NaN.
    """

    def __init__(self, split, window=5, stats=None, match_ids=None, root=PHASE2_ROOT):
        if window % 2 != 1:
            raise ValueError("window must be odd (the loss uses the middle snapshot)")
        splits = read_splits(root)
        ids = sorted(m for m, s in splits.items() if s == split) if match_ids is None else [int(m) for m in match_ids]
        wrong = [m for m in ids if splits.get(m) != split]
        if wrong:
            raise ValueError(f"matches {wrong} are not in split '{split}' according to splits.csv")
        if stats is not None and stats["window"] != window:
            raise ValueError(f"stats were fitted for window {stats['window']}, got window {window}")
        if stats is None:
            if split != "train":
                raise ValueError("stats must be fitted on train matches and passed in for valid/test")
            stats = fit_stats(ids, window, root)
        self.window, self.stats, self.split, self.match_ids = window, stats, split, ids
        self.dt_rel_rms = stats["dt_rel_rms"]
        self.matches = [MatchArrays(m, read_match(m, root), stats) for m in ids]
        self.offsets = np.cumsum([0] + [m.S for m in self.matches])
        self.categorical_indices = CATEGORICAL_INDICES
        self.categorical_features = CATEGORICAL_FEATURES

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        k = bisect.bisect_right(self.offsets, idx) - 1
        M, p = self.matches[k], int(idx - self.offsets[k])
        h = self.window // 2
        q = p + np.arange(-h, h + 1)
        ok = (q >= 0) & (q < M.S)
        qc = np.where(ok, q, 0)
        ok &= M.period[qc] == M.period[p]  # never cross the half-time boundary
        qc = np.where(ok, q, 0)
        v = M.valid[qc] & ok[:, None]  # (W, P)
        slots = np.nonzero(v.any(axis=0))[0]  # union of the players present anywhere in the window
        v = v[:, slots]

        feats = M.feats[qc][:, slots] * v[..., None]
        feats[:, :, I_DT_REL] = (_signed_log(M.t_s[qc] - M.t_s[p]) / self.dt_rel_rms)[:, None] * v
        obs = M.obs[qc][:, slots] * v[..., None]
        vis = M.vis[qc][:, slots] & v
        lab = np.where(v[..., None], M.labels[qc][:, slots], np.nan).astype(np.float32)
        a = np.nonzero(M.actor[p][slots])[0]
        return {
            "features": torch.from_numpy(feats.astype(np.float32)),
            "obs_xy": torch.from_numpy(obs.astype(np.float32)),
            "visible": torch.from_numpy(vis),
            "valid": torch.from_numpy(v),
            "labels": torch.from_numpy(lab),
            "actor_global_index_lst": torch.tensor(int(a[0]) if len(a) else -1, dtype=torch.long),
            "categorical_indices": self.categorical_indices,
            "categorical_features": self.categorical_features,
            "object_ids": [str(M.object_ids[s]) for s in slots],
            "snap_idx_window": torch.from_numpy(np.where(ok, M.snap_ids[qc], -1).astype(np.int64)),
        }


def collate_phase2(samples):
    """Stack samples, padding the player axis N to the batch maximum with the same neutral fill (valid False marks the padding)."""
    n = max(s["features"].shape[1] for s in samples)

    def pad(x, value):
        extra = n - x.shape[1]
        if extra == 0:
            return x
        shape = list(x.shape)
        shape[1] = extra
        return torch.cat([x, torch.full(shape, value, dtype=x.dtype)], dim=1)

    return {
        "features": torch.stack([pad(s["features"], 0.0) for s in samples]),
        "obs_xy": torch.stack([pad(s["obs_xy"], 0.0) for s in samples]),
        "visible": torch.stack([pad(s["visible"], False) for s in samples]),
        "valid": torch.stack([pad(s["valid"], False) for s in samples]),
        "labels": torch.stack([pad(s["labels"], float("nan")) for s in samples]),
        "actor_global_index_lst": torch.stack([s["actor_global_index_lst"] for s in samples]),
    }
