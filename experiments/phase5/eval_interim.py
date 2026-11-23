"""Phase 5 interim evaluation: current best_model.ckpt of each regime on the VALID matches of all five competitions.
Test is not read (it stays reserved for the single final evaluation). Valid of 9/10/135 (B) or 9 (A) was used for early stopping, so
those rows are mildly optimistic for that run; valid of 179 and 165 is unused by both runs.

Each run is scored with its own params.json and phase2_stats.json. Rows: has_label; visible = observed (eval_valid_ckpt.py convention).
Rule = Tier 0 rule without training (analyze_uefa.py): visible position = the 360 observation, visible velocity = clipped finite
difference, hidden position = prev->next linear path, else last sighting, else role median of the competition-9 train matches.
CI = 95% bootstrap over matches (2000 resamples) of the per-match mean position error.

usage: eval_interim.py --data_dir ROOT --out DIR --run A=RUN_DIR --run B=RUN_DIR [--split valid]
writes DIR/summary.csv, DIR/report.md, DIR/<run>_meta.json
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--run", action="append", required=True); ap.add_argument("--split", default="valid")
ap.add_argument("--save_preds", action="store_true", help="also write DIR/preds_<run>_<comp>.parquet (all evaluated columns incl. positions)")
a = ap.parse_args()
assert a.split != "test", "interim evaluation must not read test"
os.makedirs(a.out, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns_eval")
sys.path.insert(0, REPO); os.chdir(REPO)
import mlflow
from imputer import phase2 as P
from imputer.components.model import SetTransformerModel
mlflow.pytorch.autolog(disable=True)
sys.path.insert(0, os.environ.get("PHASE2_BASELINE_SRC", os.path.expanduser("~/scoutgpt_data/_phase2_baseline/src")))
from common import metric_frame

L, Wd = 105.0, 68.0
COMPS = {9: "1. Bundesliga", 10: "2. Bundesliga", 135: "Frauen-Bundesliga", 179: "3. Liga", 165: "DFB-Pokal"}
COLS = ["snap_idx", "object_id", "observed", "has_label", "align_weak", "obs_x", "obs_y", "lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay",
        "lbl_vel", "has_prev", "has_next", "prev_dt", "next_dt", "prev_x", "prev_y", "next_x", "next_y", "pos_id", "keeper", "t_s", "period", "actor", "teammate", "sb_player_id"]
sp = pd.read_csv(os.path.join(a.data_dir, "out", "splits.csv"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tr_ids = sp[(sp.competition == 9) & (sp.split == "train")].match_id
rp = pd.concat([pd.read_parquet(os.path.join(a.data_dir, "out/players", f"{m}.parquet"), columns=["pos_id", "lbl_x", "lbl_y", "has_label"]) for m in tr_ids], ignore_index=True)
rp = rp[rp.has_label]
RM = rp.groupby(rp.pos_id.fillna(0).astype(int))[["lbl_x", "lbl_y"]].median()


def load_ckpt(path):
    for _ in range(20):  # the training process may be rewriting best_model.ckpt
        m0 = os.path.getmtime(path)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print("ckpt read failed, retry:", e, flush=True); time.sleep(30); continue
        if os.path.getmtime(path) == m0:
            return ck, m0
        time.sleep(30)
    raise RuntimeError("could not read a stable checkpoint")


def collate(samples):
    b = P.collate_phase2(samples)
    b["oid"] = [s["object_ids"] for s in samples]; b["snap"] = [int(s["snap_idx_window"][2]) for s in samples]
    return b


def predict(lm, ds):
    mid_of = np.repeat([m.match_id for m in ds.matches], np.diff(ds.offsets))
    rows, i0 = [], 0
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
    pl = pd.concat([pd.read_parquet(os.path.join(a.data_dir, "out/players", f"{m}.parquet"), columns=COLS).assign(match_id=m) for m in ds.match_ids], ignore_index=True)
    pr["object_id"] = pr.object_id.astype(pl.object_id.dtype)
    d = pl.merge(pr, on=["match_id", "snap_idx", "object_id"], how="left", validate="one_to_one")
    assert d.p_x.notna().all(), "a player row has no prediction"
    d = d[d.has_label].reset_index(drop=True)
    hp, hn = d.has_prev.to_numpy(), d.has_next.to_numpy()
    pdt = np.where(hp, d.prev_dt, np.nan); ndt = np.where(hn, d.next_dt, np.nan); both = hp & hn; span = pdt + ndt
    for c in "xy":
        o, p, n = d[f"obs_{c}"].to_numpy(), d[f"prev_{c}"].to_numpy(), d[f"next_{c}"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            vc = np.where(both, (n - p) / span, np.nan); vb = (o - p) / pdt; vf = (n - o) / ndt
        d[f"r_v{c}"] = np.clip(np.where(np.isfinite(vc), vc, np.where(np.isfinite(vb), vb, np.where(np.isfinite(vf), vf, 0.0))), -12, 12)
        fb = d.pos_id.fillna(0).astype(int).map(RM[f"lbl_{c}"]).to_numpy()
        d[f"r_{c}"] = np.where(both, p + (pdt / span) * (n - p), np.where(hp, p, np.where(hn, n, fb)))
    return d


def summ(s, vis):
    g = pd.DataFrame({"g_all": ["all"] * len(s)}, index=s.index)
    tr = dict(x=s.lbl_x.values, y=s.lbl_y.values, vx=s.lbl_vx.values, vy=s.lbl_vy.values, ax=s.lbl_ax.values, ay=s.lbl_ay.values, vel=s.lbl_vel.values)
    m = metric_frame(g, tr, dict(x=s.p_x.values, y=s.p_y.values, vx=s.p_vx.values, vy=s.p_vy.values, ax=s.p_ax.values, ay=s.p_ay.values), {"overall": "g_all"}).iloc[0]
    rule = dict(x=s.obs_x.values, y=s.obs_y.values, vx=s.r_vx.values, vy=s.r_vy.values) if vis else dict(x=s.r_x.values, y=s.r_y.values)
    r = metric_frame(g, tr, rule, {"overall": "g_all"}).iloc[0]
    ta, pa = np.hypot(s.lbl_ax, s.lbl_ay), np.hypot(s.p_ax, s.p_ay)
    ts, ps = np.hypot(s.lbl_vx, s.lbl_vy), np.hypot(s.p_vx, s.p_vy)
    err = pd.Series(np.hypot(s.p_x - s.lbl_x, s.p_y - s.lbl_y).to_numpy()).groupby(s.match_id.to_numpy()).mean().to_numpy()
    bs = np.random.default_rng(0).choice(err, (2000, len(err))).mean(1)
    hi, big = ts >= 5.5, ta > 2
    return dict(n=len(s), n_matches=s.match_id.nunique(), pos_med=m.pos_med, rule_pos_med=r.pos_med, pos_match_mean=err.mean(),
                ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975), vel_rmse=m.vel_vec_rmse, rule_vel_rmse=r.get("vel_vec_rmse", np.nan),
                head_med=m.head_med_deg, acc_rmse=m.acc_vec_rmse, acc_gain=100 * (1 - m.acc_vec_rmse / np.sqrt((ta ** 2).mean())), acc_corr=m.acc_corr,
                acc_big_true=ta[big].mean(), acc_big_pred=pa[big].mean(), spd_p95=ts.quantile(.95), share_ge55=100 * hi.mean(),
                spd_hi_true=ts[hi].mean(), spd_hi_pred=ps[hi].mean())


rows = []
for spec in a.run:
    name, rdir = spec.split("=", 1)
    params = json.load(open(os.path.join(rdir, "params.json"))); stats = json.load(open(os.path.join(rdir, "phase2_stats.json")))
    ck, mt = load_ckpt(os.path.join(rdir, "best_model.ckpt"))
    lm = None
    meta = dict(run=name, dir=rdir, epoch=int(ck["epoch"]), ckpt_mtime=time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(mt)), competitions=params.get("competitions"), split=a.split)
    print(meta, flush=True)
    for c, cname in COMPS.items():
        ids = sorted(sp[(sp.competition == c) & (sp.split == a.split)].match_id.astype(int).tolist())
        ds = P.Phase2Windows(a.split, 5, stats=stats, match_ids=ids, root=a.data_dir)
        if lm is None:
            lm = SetTransformerModel(ds, params["ModelConfig"], params["OptimizerConfig"]); lm.load_state_dict(ck["state_dict"]); lm.eval(); lm.to(dev)
        t = time.time(); d = predict(lm, ds)
        trained = c in (params.get("competitions") or ())
        if a.save_preds:
            d.to_parquet(os.path.join(a.out, f"preds_{name}_{c}.parquet"), index=False)
        for var, dd in (("all", d), ("excl_weak", d[~d.align_weak])):
            for task, vis in (("visible", True), ("hidden", False)):
                q = dd[dd.observed] if vis else dd[~dd.observed]
                if len(q):
                    rows.append(dict(run=name, epoch=meta["epoch"], comp=c, competition=cname, in_train=trained, var=var, task=task, **summ(q, vis)))
        print(f"  {name} {cname}: {len(ids)} matches, {len(d):,} labelled rows, {time.time()-t:.0f}s", flush=True)
    json.dump(meta, open(os.path.join(a.out, f"{name}_meta.json"), "w"), indent=1)

S = pd.DataFrame(rows); S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
metas = [json.load(open(os.path.join(a.out, f"{s.split('=')[0]}_meta.json"))) for s in a.run]
md = [f"# Phase 5 interim evaluation ({a.split} split, test not read)", ""]
md += [f"- {m['run']}: epoch {m['epoch']} (best_model.ckpt saved {m['ckpt_mtime']}), trained on {m['competitions']}" for m in metas]
md += ["", "Valid of trained competitions was used for early stopping. CI: 95% bootstrap over matches of the per-match mean position error.", ""]
fmt = lambda x: f"{x:.2f}"
for var in ("all", "excl_weak"):
    for task in ("visible", "hidden"):
        t = S[(S["var"] == var) & (S.task == task)]
        md += [f"## {task}, {'all matches' if var == 'all' else 'align_weak matches excluded'}", "",
               "| competition | run | trained | matches | pos median (rule) m | per-match mean [CI] m | vel RMSE (rule) m/s | heading ° | acc gain % | acc corr | high-speed pred/true m/s |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.sort_values(["comp", "run"], key=lambda s: s.map({9: 0, 10: 1, 135: 2, 179: 3, 165: 4}) if s.name == "comp" else s).iterrows():
            rv = "" if np.isnan(r.rule_vel_rmse) else f" ({fmt(r.rule_vel_rmse)})"
            md.append(f"| {r.competition} | {r.run} | {'yes' if r.in_train else 'no'} | {r.n_matches} | {fmt(r.pos_med)} ({fmt(r.rule_pos_med)}) | {fmt(r.pos_match_mean)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | "
                      f"{fmt(r.vel_rmse)}{rv} | {r.head_med:.1f} | {r.acc_gain:.1f} | {fmt(r.acc_corr)} | {fmt(r.spd_hi_pred)}/{fmt(r.spd_hi_true)} |")
        md.append("")
open(os.path.join(a.out, "report.md"), "w").write("\n".join(md))
print("\n".join(md))
