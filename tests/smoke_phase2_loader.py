"""CPU smoke test for the Phase-2 loader (imputer/phase2.py) -> SetTransformerModel -> balanced_loss, on a few matches.

Run from the repo root:  .venv/bin/python -B tests/smoke_phase2_loader.py
Reads parquet files under /hpcwork/p0029161/scoutgpt/_phase2_dataset/out (read-only), 2 train + 1 valid + 1 test match.
"""
import os
import sys
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import imputer.config as config
from imputer import phase2 as P
from imputer.components import model as M

torch.manual_seed(0)
torch.set_num_threads(4)
CFG = json.load(open("imputer_params.json"))["transformer"]["ModelConfig"]
CFG["dropout"] = 0.0
W = 5
OUT = os.path.join(P.PHASE2_ROOT, "out")


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        check.failed += 1
check.failed = 0


t0 = time.time()
splits = P.read_splits()
by_split = {s: sorted(m for m, v in splits.items() if v == s) for s in ("train", "valid", "test")}
tr_ids, va_ids, te_ids = by_split["train"][:2], by_split["valid"][:1], by_split["test"][:1]
print(f"matches: train {tr_ids}, valid {va_ids}, test {te_ids}")

print("== 7. split sets (splits.csv is the source of truth)")
check(f"train/valid/test match sets disjoint ({ {s: len(v) for s, v in by_split.items()} })",
      not (set(by_split["train"]) & set(by_split["valid"])) and not (set(by_split["train"]) & set(by_split["test"])) and not (set(by_split["valid"]) & set(by_split["test"])))
check("40 / 8 / 8 matches", [len(by_split[s]) for s in ("train", "valid", "test")] == [40, 8, 8])
try:
    P.fit_stats(va_ids, W)
    ok = False
except ValueError:
    ok = True
check("fit_stats refuses a valid-split match", ok)
try:
    P.Phase2Windows("train", W, match_ids=va_ids)
    ok = False
except ValueError:
    ok = True
check("Phase2Windows('train') refuses a valid-split match", ok)
try:
    P.Phase2Windows("valid", W, match_ids=va_ids)
    ok = False
except ValueError:
    ok = True
check("valid dataset without train stats is refused", ok)

ds_tr = P.Phase2Windows("train", W, match_ids=tr_ids)
stats = ds_tr.stats
ds_va = P.Phase2Windows("valid", W, stats=stats, match_ids=va_ids)
ds_te = P.Phase2Windows("test", W, stats=stats, match_ids=te_ids)
print(f"  samples: train {len(ds_tr)}, valid {len(ds_va)}, test {len(ds_te)}; stats fit on {stats['fit_matches']}")
check("stats were fitted on the train matches only", set(stats["fit_matches"]) == set(tr_ids) and not (set(stats["fit_matches"]) & set(va_ids + te_ids)))
check("dataset match ids are all in their split", all(splits[m] == "train" for m in ds_tr.match_ids) and all(splits[m] == "valid" for m in ds_va.match_ids) and all(splits[m] == "test" for m in ds_te.match_ids))

print("== 2. shapes and dtypes of a collated batch")
dl = DataLoader(ds_va, batch_size=8, shuffle=False, collate_fn=P.collate_phase2)
batch = next(iter(dl))
B, Wn, N = batch["features"].shape[:3]
print(f"  B={B} W={Wn} N={N} F={P.F_DIM}")
want = {"features": ((B, W, N, P.F_DIM), torch.float32), "obs_xy": ((B, W, N, 2), torch.float32), "visible": ((B, W, N), torch.bool),
        "valid": ((B, W, N), torch.bool), "labels": ((B, W, N, 6), torch.float32), "actor_global_index_lst": ((B,), torch.int64)}
check(f"exactly the keys {sorted(want)}", sorted(batch) == sorted(want))
for k, (shape, dtype) in want.items():
    check(f"{k}: {tuple(batch[k].shape)} {batch[k].dtype} == {shape} {dtype}", tuple(batch[k].shape) == shape and batch[k].dtype == dtype)

print("== 1. end to end: loader -> SetTransformerModel -> balanced_loss -> backward")
lm = M.SetTransformerModel(ds_tr, dict(CFG), {"optimizer_params": {"lr": 1e-4}})
lm.train()
dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, collate_fn=P.collate_phase2)
b = next(iter(dl_tr))
mae, loss, eu, cos = lm.step(b)
loss.backward()
grads_ok = all(torch.isfinite(p.grad).all().item() for p in lm.parameters() if p.grad is not None)
print(f"  loss {loss.item():.4f}  euclid {eu.item():.2f} m  cos {cos.item():.3f}  mae {[round(x, 3) for x in mae.tolist()]}")
check("loss, metrics finite", all(torch.isfinite(x).item() for x in (loss, eu, cos)) and torch.isfinite(mae).all().item())
check("all gradients finite", grads_ok)
check("model output has no NaN (hidden obs_xy / padding do not propagate NaN)", torch.isfinite(lm.model.eval()(b["features"], b["obs_xy"], b["visible"], b["valid"])).all().item())
check("no NaN in features / obs_xy", torch.isfinite(b["features"]).all().item() and torch.isfinite(b["obs_xy"]).all().item())

print("== 3. no forbidden column reaches features")
used = {c for v in P.FEATURE_SOURCES.values() for c in v} | set(P.OBS_SOURCES)
print(f"  columns feeding features/obs_xy/visible: {sorted(used)}")
check(f"feature/obs sources intersect FORBIDDEN_COLUMNS in {sorted(used & P.FORBIDDEN_COLUMNS)}", not (used & P.FORBIDDEN_COLUMNS))
check("explicit deny-list contains every lbl_*, in_tracking, has_label, sb_player_id, align_weak, split, match_id, dfl_match_id",
      {"lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay", "lbl_vel", "lbl_acc", "lbl_heading", "in_tracking", "has_label", "sb_player_id",
       "align_weak", "split", "match_id", "dfl_match_id"} <= set(P.FORBIDDEN_COLUMNS))
df = P.read_match(tr_ids[0])
check("read columns exclude in_tracking / sb_player_id / team_id / jersey / lbl_vel / lbl_acc / lbl_heading entirely",
      not (set(df.columns) & {"in_tracking", "sb_player_id", "team_id", "jersey", "lbl_vel", "lbl_acc", "lbl_heading", "match_id", "split", "align_weak"}))
rng = np.random.default_rng(0)
df2 = df.copy()
for c in P.LABEL_COLUMNS:
    df2[c] = rng.permutation(df2[c].values)
df2["has_label"] = ~df2["has_label"]
for c in ("in_tracking", "sb_player_id", "team_id", "jersey", "align_weak", "split", "match_id"):
    if c in df2:
        df2[c] = rng.permutation(df2[c].values)
A, A2 = P.MatchArrays(tr_ids[0], df, stats), P.MatchArrays(tr_ids[0], df2, stats)
same = all(np.array_equal(getattr(A, k), getattr(A2, k)) for k in ("feats", "obs", "vis", "valid", "actor"))
check("scrambling every label column / has_label / in_tracking / ids leaves features, obs_xy, visible, valid bit-identical", same)
check("control: the same scrambling does change the labels", not np.array_equal(A.labels, A2.labels, equal_nan=True))

print("== -1.0 sentinel and NaN handling")
sent = ~df["has_prev"].values
sp_, sl_ = np.unique(df["snap_idx"].values, return_inverse=True)[1], np.unique(df["object_id"].values, return_inverse=True)[1]
f_rows = A.feats[sp_, sl_]
check("has_prev False rows: prev_dt_log, prev_x, prev_y all exactly 0 (the -1.0 sentinel is not a duration)",
      bool((f_rows[sent][:, [P.I_PREV_DT, P.FEATURES.index('prev_x'), P.FEATURES.index('prev_y')]] == 0).all()) and sent.sum() > 0)
nsent = ~df["has_next"].values
check("has_next False rows: next_dt_log, next_x, next_y all exactly 0",
      bool((f_rows[nsent][:, [P.I_NEXT_DT, P.FEATURES.index('next_x'), P.FEATURES.index('next_y')]] == 0).all()) and nsent.sum() > 0)
check("has_prev True rows: prev_dt_log is the standardised log1p (mean ~0 over this match's rows)", abs(f_rows[~sent][:, P.I_PREV_DT].mean()) < 0.5)
check("hidden rows: obs_xy is exactly 0 (no NaN upstream)", bool((A.obs[sp_, sl_][~df["observed"].values] == 0).all()) and bool(np.isfinite(A.obs).all()))

print("== 4. windows never straddle a period or match boundary (checked against snapshots/<match>.parquet)")
snaps = pd.read_parquet(os.path.join(OUT, "snapshots", f"{va_ids[0]}.parquet"), columns=["snap_idx", "period", "t_s"]).sort_values("snap_idx")
per = dict(zip(snaps.snap_idx, snaps.period))
tsd = dict(zip(snaps.snap_idx, snaps.t_s))
bad, n_empty_steps, n_checked = 0, 0, 0
for i in range(len(ds_va)):
    s = ds_va[i]
    w = s["snap_idx_window"].numpy()
    tgt = w[W // 2]
    got = w[w >= 0]
    n_checked += 1
    n_empty_steps += int((w < 0).sum())
    if any(per[x] != per[tgt] for x in got) or tgt != snaps.snap_idx.values[i]:
        bad += 1
    # consecutive and ordered: present steps are the contiguous snapshots tgt-h..tgt+h that exist in the same period
    exp = [snaps.snap_idx.values[j] if 0 <= j < len(snaps) and per[snaps.snap_idx.values[j]] == per[tgt] else -1 for j in range(i - W // 2, i + W // 2 + 1)]
    if list(w) != exp:
        bad += 1
print(f"  {n_checked} windows checked, {n_empty_steps} empty (edge) timesteps")
check("0 windows contain a snapshot from the other period or out of order", bad == 0)
p2 = int(np.argmax(snaps.period.values == 2))
first_p2, last_p1 = ds_va[p2], ds_va[p2 - 1]
check("first snapshot of period 2: the two preceding timesteps are empty", list(first_p2["snap_idx_window"].numpy()[:2]) == [-1, -1] and not first_p2["valid"][:2].any())
check("last snapshot of period 1: the two following timesteps are empty", list(last_p1["snap_idx_window"].numpy()[3:]) == [-1, -1] and not last_p1["valid"][3:].any())
check("first snapshot of the match: two empty timesteps before it", list(ds_va[0]["snap_idx_window"].numpy()[:2]) == [-1, -1])
check("last snapshot of the match: two empty timesteps after it", list(ds_va[len(ds_va) - 1]["snap_idx_window"].numpy()[3:]) == [-1, -1])
check("the middle timestep is always the target snapshot and is non-empty", all(ds_va[i]["valid"][W // 2].any().item() for i in range(0, len(ds_va), 50)))

print("== 5/6. valid is False exactly where a player is absent; labels are NaN exactly where ~has_label (checked against the parquet)")
pl = pd.read_parquet(os.path.join(OUT, "players", f"{va_ids[0]}.parquet"), columns=["snap_idx", "object_id", "has_label", "observed", "obs_x", "lbl_x", "teammate", "actor"])
key = {(s_, o_): (hl, ob, ox) for s_, o_, hl, ob, ox in zip(pl.snap_idx.values, pl.object_id.values, pl.has_label.values, pl.observed.values, pl.obs_x.values)}
bad_valid = bad_lab = bad_vis = bad_obs = 0
n_tokens = n_hidden_nolabel = 0
for i in list(range(0, len(ds_va), 7)) + [p2 - 1, p2, 0, len(ds_va) - 1]:
    s = ds_va[i]
    w, objs, v, lab, vis, obs = s["snap_idx_window"].numpy(), s["object_ids"], s["valid"].numpy(), s["labels"].numpy(), s["visible"].numpy(), s["obs_xy"].numpy()
    for t in range(W):
        for n, o in enumerate(objs):
            present = w[t] >= 0 and (w[t], o) in key
            bad_valid += int(v[t, n] != present)
            if present:
                hl, ob, ox = key[(w[t], o)]
                bad_lab += int(np.isnan(lab[t, n]).all() == hl or np.isnan(lab[t, n]).any() != (not hl))
                bad_vis += int(vis[t, n] != ob)
                bad_obs += int(ob and abs(obs[t, n, 0] - ox / config.field_length) > 1e-6)
                n_hidden_nolabel += int(not hl)
            else:
                bad_lab += int(not np.isnan(lab[t, n]).all())
                bad_vis += int(vis[t, n])
            n_tokens += 1
print(f"  {n_tokens} tokens checked ({n_hidden_nolabel} rows without label)")
check("valid == (row exists in the parquet for that snapshot and player), 0 mismatches", bad_valid == 0)
check("labels all-NaN exactly where ~has_label or absent, fully finite where has_label, 0 mismatches", bad_lab == 0)
check("visible == observed, obs_xy == obs_x/105 for visible players, 0 mismatches", bad_vis == 0 and bad_obs == 0)
check("a player absent in some timesteps of a window exists in the data (the substitution case is exercised)",
      any((~ds_va[i]["valid"]).any(dim=0).logical_and(ds_va[i]["valid"].any(dim=0)).any().item() for i in range(len(ds_va))))

print("== 5b. padding to the batch maximum does not change the output")
lm.eval()
sa, sb_ = None, None
ns = [ds_va[i]["features"].shape[1] for i in range(len(ds_va))]
ia, ib = int(np.argmin(ns)), int(np.argmax(ns))
sa, sb_ = ds_va[ia], ds_va[ib]
print(f"  sample N={sa['features'].shape[1]} alone vs padded to N={sb_['features'].shape[1]}")
alone = P.collate_phase2([sa])
padded = P.collate_phase2([sa, sb_])
with torch.no_grad():
    o1 = lm.model(alone["features"], alone["obs_xy"], alone["visible"], alone["valid"])
    o2 = lm.model(padded["features"], padded["obs_xy"], padded["visible"], padded["valid"])[:1, :, : o1.shape[2]]
vm = alone["valid"][0]  # 값이 의미 있는 token만 비교 (valid False 위치의 출력은 loss에서 쓰이지 않음)
d = (o1 - o2)[:, vm].abs().max().item()
d_all = (o1 - o2).abs().max().item()
print(f"  max diff at valid tokens {d:.1e}; at invalid tokens (unused by the loss) {d_all:.1e}")
check(f"padded slots (valid False) leave the outputs at valid tokens unchanged (max diff {d:.1e} < 1e-4)", d < 1e-4)
check("variable N: the batch maximum exceeds the smallest sample's N", sb_["features"].shape[1] >= sa["features"].shape[1])

print("== train-only normalisation statistics (these 2 train matches)")
print("  " + json.dumps({k: v for k, v in stats.items() if k != "fit_matches"}))
print(f"done in {time.time() - t0:.1f}s, failures: {check.failed}")
sys.exit(1 if check.failed else 0)
