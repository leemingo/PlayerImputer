"""Valid-split evaluation of one checkpoint with the Phase 2 baseline metric_frame (same rows: has_label, visible = observed),
plus acceleration / speed summaries. Same procedure as ref_epoch65/eval_valid.py. Test is not touched.

usage: eval_valid_ckpt.py --ckpt PATH --out DIR [--tag NAME]
writes DIR/metrics_valid.csv, DIR/key_valid.json, DIR/preds_valid.parquet
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--split", default="valid", help="split name in <PHASE2_DATA>/out/splits.csv")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("PHASE2_DATA", os.path.expanduser("~/scoutgpt_data/_phase2_dataset"))
RUN1 = os.path.join(REPO, "stores/transformer/001")
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns_eval")
sys.path.insert(0, REPO); os.chdir(REPO)
import mlflow
from imputer import phase2 as P
from imputer.components.model import SetTransformerModel
mlflow.pytorch.autolog(disable=True)
sys.path.insert(0, os.environ.get("PHASE2_BASELINE_SRC", os.path.expanduser("~/scoutgpt_data/_phase2_baseline/src")))  # Phase 2b common.py (metric_frame)
from common import metric_frame, bucket, EDGES, LAB

L, Wd = 105.0, 68.0
params = json.load(open(os.path.join(RUN1, "params.json")))
stats = json.load(open(os.path.join(RUN1, "phase2_stats.json")))
ds = P.Phase2Windows(a.split, 5, stats=stats, root=DATA)
lm = SetTransformerModel(ds, params["ModelConfig"], params["OptimizerConfig"])
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
lm.load_state_dict(ck["state_dict"]); lm.eval()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); lm.to(dev)
print("ckpt", a.ckpt, "epoch", ck["epoch"], "device", dev, flush=True)


def collate(samples):
    b = P.collate_phase2(samples)
    b["oid"] = [s["object_ids"] for s in samples]
    b["snap"] = [int(s["snap_idx_window"][2]) for s in samples]
    return b


mid_of = np.repeat([m.match_id for m in ds.matches], np.diff(ds.offsets))
rows, i0, t = [], 0, time.time()
with torch.no_grad():
    for b in DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=4):
        out = lm(b["features"].to(dev), b["obs_xy"].to(dev), b["visible"].to(dev), b["valid"].to(dev))[:, 2].cpu().numpy()
        val = b["valid"][:, 2].numpy()
        for j in range(out.shape[0]):
            n = len(b["oid"][j]); ok = val[j, :n]; o = out[j, :n][ok]
            rows.append(pd.DataFrame({"match_id": mid_of[i0 + j], "snap_idx": b["snap"][j], "object_id": np.array(b["oid"][j])[ok],
                                      "p_vx": o[:, 0], "p_vy": o[:, 1], "p_x": o[:, 2] * L, "p_y": o[:, 3] * Wd, "p_ax": o[:, 4], "p_ay": o[:, 5]}))
        i0 += out.shape[0]
pr = pd.concat(rows, ignore_index=True)
cols = ["snap_idx", "object_id", "period", "t_s", "observed", "has_label", "align_weak", "obs_x", "obs_y", "lbl_x", "lbl_y", "lbl_vx", "lbl_vy",
        "lbl_ax", "lbl_ay", "lbl_vel", "has_prev", "has_next", "prev_dt", "next_dt", "keeper", "teammate", "actor", "is_home"]
pl = pd.concat([pd.read_parquet(os.path.join(DATA, "out/players", f"{m}.parquet"), columns=cols).assign(match_id=m) for m in ds.match_ids], ignore_index=True)
pr["object_id"] = pr["object_id"].astype(pl["object_id"].dtype)
d = pl.merge(pr, on=["match_id", "snap_idx", "object_id"], how="left", validate="one_to_one")
assert d.p_x.notna().all(), "a player row has no prediction"
d.to_parquet(os.path.join(a.out, "preds_valid.parquet"), index=False)
print(f"forward+join {time.time()-t:.1f}s, rows {len(d):,}", flush=True)

d = d[d.has_label].reset_index(drop=True)
d["prev_dt"] = np.where(d.has_prev, d.prev_dt, np.nan); d["next_dt"] = np.where(d.has_next, d.next_dt, np.nan)
d["span"] = d.prev_dt + d.next_dt
both = d.has_prev & d.has_next
d["g_all"] = "all"
d["g_span"] = np.where(both, bucket(d.span, EDGES, LAB).astype(object), np.where(d.has_prev, "prev_only", np.where(d.has_next, "next_only", "neither")))
d["g_prev"] = np.where(d.has_prev, bucket(d.prev_dt, EDGES, LAB).astype(object), "none")
d["g_keeper"] = np.where(d.keeper, "GK", "outfield")
groups = dict(overall="g_all", span="g_span", prev_dt="g_prev", keeper="g_keeper")
res, key = [], {"tag": a.tag, "ckpt": a.ckpt, "epoch": int(ck["epoch"])}
for tname, sel in (("visible", d.observed.values), ("hidden", ~d.observed.values)):
    for var, m2 in (("all", np.ones(len(d), bool)), ("excl8", ~d.align_weak.values)):
        s = d[sel & m2].reset_index(drop=True)
        tr = dict(x=s.lbl_x.values, y=s.lbl_y.values, vx=s.lbl_vx.values, vy=s.lbl_vy.values, ax=s.lbl_ax.values, ay=s.lbl_ay.values, vel=s.lbl_vel.values)
        p = dict(x=s.p_x.values, y=s.p_y.values, vx=s.p_vx.values, vy=s.p_vy.values, ax=s.p_ax.values, ay=s.p_ay.values)
        r = metric_frame(s, tr, p, groups); r.insert(0, "var", var); r.insert(0, "task", tname); res.append(r)
        if var == "all":
            o = r[r.group_kind == "overall"].iloc[0]
            ta, pa = np.hypot(s.lbl_ax, s.lbl_ay), np.hypot(s.p_ax, s.p_ay)
            ts, ps = np.hypot(s.lbl_vx, s.lbl_vy), np.hypot(s.p_vx, s.p_vy)
            big = ta > 2
            key[tname] = dict(n=len(s), pos_med=o.pos_med, pos_mean=o.pos_mean, vel_vec_rmse=o.vel_vec_rmse, spd_rmse=o.spd_rmse, r2=o.r2,
                              head_med_deg=o.head_med_deg, acc_vec_rmse=o.acc_vec_rmse, acc_corr=o.acc_corr,
                              acc_zero_rmse=float(np.sqrt((ta ** 2).mean())), acc_gain_vs_zero_pct=float(100 * (1 - o.acc_vec_rmse / np.sqrt((ta ** 2).mean()))),
                              acc_true_mag_mean=float(ta.mean()), acc_pred_mag_mean=float(pa.mean()),
                              acc_big_true_mean=float(ta[big].mean()), acc_big_pred_mean=float(pa[big].mean()),
                              spd_hi_true_mean=float(ts[ts >= 5.5].mean()), spd_hi_pred_mean=float(ps[ts >= 5.5].mean()))
out = pd.concat(res, ignore_index=True)
out.to_csv(os.path.join(a.out, "metrics_valid.csv"), index=False)
json.dump(key, open(os.path.join(a.out, "key_valid.json"), "w"), indent=1, default=float)
print(json.dumps(key, indent=1, default=float))
