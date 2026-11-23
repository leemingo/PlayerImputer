"""Phase 6 T3: Phase2Windows with per-player SkillCorner features appended to sample["features"].

Source: a table with columns match_id, sb_player_id and the six values in SC_COLS (pm_* = this player-match, mt_* = see the table's
builder). Each player's slot gets the six values standardised with train-only mean / std plus two flags (has_sc_pm, has_sc_mt);
missing values -> 0 with flag 0. The vector is constant over the window and zeroed where the slot is not valid (neutral fill, as for
all features). sb_player_id is read from the players parquet as a join key only and never enters the features.
sc_dropout (training only): with this probability per sample, all SC values and flags of the whole window are zeroed.
sc_off: zero them for every sample (evaluation with SkillCorner forced off).
"""
import os
import numpy as np
import pandas as pd
import torch
from imputer import phase2 as P

SC_COLS = ["pm_psv99", "pm_peak_velocity", "pm_total_metersperminute_full_all", "mt_psv99", "mt_peak_velocity", "mt_total_metersperminute_full_all"]
SC_DIM = len(SC_COLS) + 2


def fit_sc_stats(sc, train_ids):
    t = sc[sc.match_id.isin(train_ids)]
    return {c: {"mean": float(t[c].mean()), "std": float(t[c].std())} for c in SC_COLS} | {"fit_matches": sorted(int(m) for m in train_ids), "fit_rows": int(len(t))}


class ScWindows(P.Phase2Windows):
    def __init__(self, split, window=5, stats=None, match_ids=None, root=P.PHASE2_ROOT, sc_path=None, sc_stats=None, sc_dropout=0.0, sc_off=False):
        super().__init__(split, window, stats=stats, match_ids=match_ids, root=root)
        sc = pd.read_parquet(sc_path)
        self.sc_stats = sc_stats if sc_stats is not None else fit_sc_stats(sc, [m.match_id for m in self.matches])
        self.sc_dropout, self.sc_off = sc_dropout, sc_off
        self.sc_arr, self.sc_slot = [], []
        n_hit = n_tot = 0
        for M in self.matches:
            key = pd.read_parquet(os.path.join(root, "out", "players", f"{M.match_id}.parquet"), columns=["object_id", "sb_player_id"]).drop_duplicates("object_id")
            key = pd.DataFrame({"object_id": M.object_ids}).merge(key, on="object_id", how="left")
            r = key.merge(sc[sc.match_id == M.match_id], on="sb_player_id", how="left")
            assert len(r) == len(M.object_ids)
            arr = np.zeros((len(r), SC_DIM), np.float32)
            for j, c in enumerate(SC_COLS):
                arr[:, j] = np.nan_to_num(((r[c] - self.sc_stats[c]["mean"]) / self.sc_stats[c]["std"]).to_numpy(float), nan=0.0)
            arr[:, 6] = r[SC_COLS[:3]].notna().all(axis=1).to_numpy()
            arr[:, 7] = r[SC_COLS[3:]].notna().all(axis=1).to_numpy()
            self.sc_arr.append(arr); self.sc_slot.append({str(o): i for i, o in enumerate(M.object_ids)})
            n_hit += int(arr[:, 6].sum()); n_tot += len(arr)
        self.sc_coverage = n_hit / max(n_tot, 1)

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        k = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        rows = [self.sc_slot[k][o] for o in s["object_ids"]]
        v = s["valid"].numpy()[..., None].astype(np.float32)  # (W, N, 1)
        sc = np.broadcast_to(self.sc_arr[k][rows][None], (v.shape[0], len(rows), SC_DIM)) * v
        if self.sc_off or (self.sc_dropout > 0 and np.random.random() < self.sc_dropout):
            sc = np.zeros_like(sc)
        s["features"] = torch.cat([s["features"], torch.from_numpy(np.ascontiguousarray(sc, dtype=np.float32))], dim=-1)
        return s
