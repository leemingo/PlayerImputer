"""Phase 6 evaluation of loss / output variants (adapted from experiments/phase5/eval_interim.py): best_model.ckpt of each run on the
VALID matches of all five competitions (test is not read), each run with its own params.json / phase2_stats.json. The model class
comes from params.json "variant": prob (T1, ProbModel), lds (T2, LDSModel, weights from <run>/lds_weights.json), absent = plain
SetTransformerModel (e.g. the Phase 5 A baseline). For prob runs the means are the point predictions; the predicted std of every
component is exported (s_vx .. s_ay, label units with x, y in metres) and the summary adds Gaussian interval coverage: 2-D (pos, vel,
acc) at 50/80/90 % (chi-square, 2 dof, independent components) and the mean predicted std per group.
Rows: has_label; visible = observed. Rule = Tier 0 rule (see eval_interim.py). CI = 95% bootstrap over matches.

sc runs (T3) use sc_data.ScWindows with the run's sc_stats.json; --sc_off zeroes all SkillCorner features and flags.
Other builds (e.g. the UEFA build): --data_dir <build> --split uefa --comp_col comp --comps 16,35,353; the Tier 0 role medians always
come from the competition-9 train matches of --rule_data_dir (default: the Phase 5 multi-league build).

usage: eval_variant.py --data_dir ROOT --out DIR --run NAME=RUN_DIR [--run ...] [--split valid] [--comps 9,10,135,179,165]
                       [--comp_col competition] [--rule_data_dir DIR] [--sc_off] [--sc_path P] [--save_preds]
writes DIR/summary.csv, DIR/report.md, DIR/<run>_meta.json
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--run", action="append", required=True); ap.add_argument("--split", default="valid")
ap.add_argument("--comps", default="9,10,135,179,165"); ap.add_argument("--comp_col", default="competition")
ap.add_argument("--rule_data_dir", default=os.path.expanduser("~/scoutgpt_data/_phase2_multileague"))
ap.add_argument("--sc_off", action="store_true"); ap.add_argument("--sc_path", default=None, help="default: sc_path in the run's sc_stats.json")
ap.add_argument("--save_preds", action="store_true", help="also write DIR/preds_<run>_<comp>.parquet (all evaluated columns incl. positions)")
a = ap.parse_args()
assert a.split != "test", "interim evaluation must not read test"
os.makedirs(a.out, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns_eval")
sys.path.insert(0, REPO); os.chdir(REPO)
import mlflow
from imputer import phase2 as P
import imputer.components.model as MM
from imputer.components.model import SetTransformerModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import ProbModel, LDSModel
from sc_data import ScWindows
mlflow.pytorch.autolog(disable=True)
sys.path.insert(0, os.environ.get("PHASE2_BASELINE_SRC", os.path.expanduser("~/scoutgpt_data/_phase2_baseline/src")))
from common import metric_frame

L, Wd = 105.0, 68.0
NAMES = {9: "1. Bundesliga", 10: "2. Bundesliga", 135: "Frauen-Bundesliga", 179: "3. Liga", 165: "DFB-Pokal", 16: "Champions League", 35: "Europa League", 353: "Conference League"}
COMPS = {int(c): NAMES.get(int(c), str(c)) for c in a.comps.split(",")}
ORDER = {c: i for i, c in enumerate(COMPS)}
COLS = ["snap_idx", "object_id", "observed", "has_label", "align_weak", "obs_x", "obs_y", "lbl_x", "lbl_y", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay",
        "lbl_vel", "has_prev", "has_next", "prev_dt", "next_dt", "prev_x", "prev_y", "next_x", "next_y", "pos_id", "keeper", "t_s", "period", "actor", "teammate", "sb_player_id"]
sp = pd.read_csv(os.path.join(a.data_dir, "out", "splits.csv"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rsp = pd.read_csv(os.path.join(a.rule_data_dir, "out", "splits.csv"))
tr_ids = rsp[(rsp.competition == 9) & (rsp.split == "train")].match_id
rp = pd.concat([pd.read_parquet(os.path.join(a.rule_data_dir, "out/players", f"{m}.parquet"), columns=["pos_id", "lbl_x", "lbl_y", "has_label"]) for m in tr_ids], ignore_index=True)
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
            if out.shape[-1] == 12:  # prob: log-variance in TARGET_SCALE units -> std in label units (x, y in metres)
                out[..., 6:] = np.exp(0.5 * out[..., 6:]) * SD_SCALE
            val = b["valid"][:, 2].numpy()
            for j in range(out.shape[0]):
                n = len(b["oid"][j]); ok = val[j, :n]; o = out[j, :n][ok]
                r_ = pd.DataFrame({"match_id": mid_of[i0 + j], "snap_idx": b["snap"][j], "object_id": np.array(b["oid"][j])[ok],
                                   "p_vx": o[:, 0], "p_vy": o[:, 1], "p_x": o[:, 2] * L, "p_y": o[:, 3] * Wd, "p_ax": o[:, 4], "p_ay": o[:, 5]})
                if o.shape[1] == 12:
                    for k, c in enumerate(("vx", "vy", "x", "y", "ax", "ay")):
                        r_[f"s_{c}"] = o[:, 6 + k]
                rows.append(r_)
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


SD_SCALE = np.array([MM.TARGET_SCALE[c] * {"x": L, "y": Wd}.get(c, 1.0) for c in ("vx", "vy", "x", "y", "ax", "ay")], dtype=np.float32)
CHI2 = {50: 1.3863, 80: 3.2189, 90: 4.6052}  # chi-square quantiles, 2 dof


def coverage(s):
    out = {}
    for g, (cx, cy, tx, ty, px, py) in dict(pos=("x", "y", "lbl_x", "lbl_y", "p_x", "p_y"), vel=("vx", "vy", "lbl_vx", "lbl_vy", "p_vx", "p_vy"),
                                            acc=("ax", "ay", "lbl_ax", "lbl_ay", "p_ax", "p_ay")).items():
        d2 = ((s[px] - s[tx]) / s[f"s_{cx}"]) ** 2 + ((s[py] - s[ty]) / s[f"s_{cy}"]) ** 2
        for q, c in CHI2.items():
            out[f"cov{q}_{g}"] = 100 * float((d2 <= c).mean())
        out[f"sd_{g}"] = float(np.sqrt((s[f"s_{cx}"] ** 2 + s[f"s_{cy}"] ** 2) / 2).mean())
    return out


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
                acc_big_true=ta[big].mean(), acc_big_pred=pa[big].mean(), spd_p95=ts.quantile(.95), share_ge55=100 * hi.mean(), share_ge55_pred=100 * (ps >= 5.5).mean(),
                spd_hi_true=ts[hi].mean(), spd_hi_pred=ps[hi].mean(), **(coverage(s) if "s_x" in s else {}))


rows = []
for spec in a.run:
    name, rdir = spec.split("=", 1)
    params = json.load(open(os.path.join(rdir, "params.json"))); stats = json.load(open(os.path.join(rdir, "phase2_stats.json")))
    ck, mt = load_ckpt(os.path.join(rdir, "best_model.ckpt"))
    lm = None
    meta = dict(run=name, variant=params.get("variant"), sc_off=bool(a.sc_off and params.get("variant") == "sc"), data_dir=a.data_dir, dir=rdir, epoch=int(ck["epoch"]), ckpt_mtime=time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(mt)), competitions=params.get("competitions"), split=a.split)
    print(meta, flush=True)
    for c, cname in COMPS.items():
        ids = sorted(sp[(sp[a.comp_col] == c) & (sp.split == a.split)].match_id.astype(int).tolist())
        if params.get("variant") == "sc":
            scs = json.load(open(os.path.join(rdir, "sc_stats.json")))
            ds = ScWindows(a.split, 5, stats=stats, match_ids=ids, root=a.data_dir, sc_path=a.sc_path or scs["sc_path"], sc_stats=scs, sc_off=a.sc_off)
        else:
            ds = P.Phase2Windows(a.split, 5, stats=stats, match_ids=ids, root=a.data_dir)
        if lm is None:
            v_ = params.get("variant")
            if v_ == "prob":
                lm = ProbModel(ds, params["ModelConfig"], params["OptimizerConfig"])
            elif v_ == "lds":
                lm = LDSModel(ds, params["ModelConfig"], params["OptimizerConfig"], lds=json.load(open(os.path.join(rdir, "lds_weights.json"))))
            else:
                lm = SetTransformerModel(ds, params["ModelConfig"], params["OptimizerConfig"])
            lm.load_state_dict(ck["state_dict"]); lm.eval(); lm.to(dev)
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
md = [f"# Phase 6 variant evaluation ({a.split} split, test not read)", ""]
md += [f"- {m['run']}: epoch {m['epoch']} (best_model.ckpt saved {m['ckpt_mtime']}), trained on {m['competitions']}" for m in metas]
md += ["", "Valid of trained competitions was used for early stopping. CI: 95% bootstrap over matches of the per-match mean position error.", ""]
fmt = lambda x: f"{x:.2f}"
for var in ("all", "excl_weak"):
    for task in ("visible", "hidden"):
        t = S[(S["var"] == var) & (S.task == task)]
        md += [f"## {task}, {'all matches' if var == 'all' else 'align_weak matches excluded'}", "",
               "| competition | run | trained | matches | pos median (rule) m | per-match mean [CI] m | vel RMSE (rule) m/s | heading ° | acc gain % | acc corr | high-speed pred/true m/s |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.sort_values(["comp", "run"], key=lambda s: s.map(ORDER) if s.name == "comp" else s).iterrows():
            rv = "" if np.isnan(r.rule_vel_rmse) else f" ({fmt(r.rule_vel_rmse)})"
            md.append(f"| {r.competition} | {r.run} | {'yes' if r.in_train else 'no'} | {r.n_matches} | {fmt(r.pos_med)} ({fmt(r.rule_pos_med)}) | {fmt(r.pos_match_mean)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | "
                      f"{fmt(r.vel_rmse)}{rv} | {r.head_med:.1f} | {r.acc_gain:.1f} | {fmt(r.acc_corr)} | {fmt(r.spd_hi_pred)}/{fmt(r.spd_hi_true)} |")
        md.append("")
if "cov90_pos" in S:
    t = S[(S["var"] == "all") & S.cov90_pos.notna()]
    md += ["## predicted-interval coverage (prob runs; nominal 50 / 80 / 90 %)", "",
           "| competition | run | task | pos cov % | pos sd m | vel cov % | vel sd m/s | acc cov % | acc sd m/s² |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values(["comp", "run", "task"], key=lambda s: s.map(ORDER) if s.name == "comp" else s).iterrows():
        md.append(f"| {r.competition} | {r.run} | {r.task} | " + " | ".join(f"{r[f'cov50_{g}']:.0f}/{r[f'cov80_{g}']:.0f}/{r[f'cov90_{g}']:.0f} | {r[f'sd_{g}']:.2f}" for g in ("pos", "vel", "acc")) + " |")
    md.append("")
open(os.path.join(a.out, "report.md"), "w").write("\n".join(md))
print("\n".join(md))
