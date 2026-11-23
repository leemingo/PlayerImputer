"""Phase 6: train one loss / output variant on Phase 5 regime A (competition 9 train, early stopping on competition 9 valid).

usage: train_variant.py --variant prob|lds|sc --data_dir ROOT --out DIR [--w_acc 2.0] [--seed 0] [--smoke N]
Everything else as experiments/phase5/train_regime.py regime A (imputer_params.json transformer block, train._phase2_trainer:
EarlyStopping patience 5, min_epochs 100, max_epochs 200, one best_model.ckpt). Normalisation statistics (and for lds the weight
tables, for sc the SkillCorner mean / std) are fitted on the train matches only and saved to DIR. sc = baseline loss with SkillCorner
features appended to the inputs (sc_data.ScWindows, --sc_path, SkillCorner dropout 0.25 per training sample). Test is not read. --smoke N: run N train batches on the current device,
print the losses and exit (no training run, nothing saved except the stats / weights).
"""
import os, sys, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=("prob", "lds", "sc"), required=True)
ap.add_argument("--sc_path", default=os.path.expanduser("~/scoutgpt_data/_phase5_context/sc_features.parquet"))
ap.add_argument("--sc_dropout", type=float, default=0.25)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--w_acc", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(REPO)

import numpy as np, torch, mlflow, pandas as pd, pytorch_lightning as pl
from imputer import phase2 as P
import imputer.components.model as M
import train as T
from variants import ProbModel, LDSModel, fit_lds_weights
from sc_data import ScWindows, SC_DIM

mlflow.pytorch.autolog(disable=True)
M.LOSS_W_ACC = a.w_acc
COMPS = (9,)
params = json.load(open(os.path.join(REPO, "imputer_params.json")))["transformer"]
params["save_path"] = a.out
sp = pd.read_csv(os.path.join(a.data_dir, "out", "splits.csv"))
ids = {s: sorted(sp[sp.competition.isin(COMPS) & (sp.split == s)].match_id.astype(int).tolist()) for s in ("train", "valid")}
json.dump(dict(params, variant=a.variant, regime="A", competitions=COMPS, w_acc=a.w_acc, seed=a.seed, data=a.data_dir, match_ids=ids),
          open(os.path.join(a.out, "params.json"), "w"), indent=1)
pl.seed_everything(a.seed, workers=True)

if a.variant == "sc":
    tr = ScWindows("train", 5, match_ids=ids["train"], root=a.data_dir, sc_path=a.sc_path, sc_dropout=a.sc_dropout)
    json.dump(dict(tr.sc_stats, sc_path=a.sc_path, sc_dropout=a.sc_dropout), open(os.path.join(a.out, "sc_stats.json"), "w"), indent=1)
    va = ScWindows("valid", 5, stats=tr.stats, match_ids=ids["valid"], root=a.data_dir, sc_path=a.sc_path, sc_stats=tr.sc_stats)
    print(f"SkillCorner: player-slot coverage train {tr.sc_coverage:.3f} valid {va.sc_coverage:.3f}; feature dim {tr[0]['features'].shape[-1]} (+{SC_DIM})", flush=True)
else:
    tr = P.Phase2Windows("train", 5, match_ids=ids["train"], root=a.data_dir)
    va = P.Phase2Windows("valid", 5, stats=tr.stats, match_ids=ids["valid"], root=a.data_dir)
json.dump(tr.stats, open(os.path.join(a.out, T.PHASE2_STATS_FILE), "w"), indent=1)

if a.variant == "lds":
    lab = pd.concat([pd.read_parquet(os.path.join(a.data_dir, "out/players", f"{m}.parquet"), columns=["has_label", "lbl_vx", "lbl_vy", "lbl_ax", "lbl_ay"])
                     for m in ids["train"]], ignore_index=True)
    lab = lab[lab.has_label]
    lds = fit_lds_weights(np.hypot(lab.lbl_vx, lab.lbl_vy).to_numpy(), np.hypot(lab.lbl_ax, lab.lbl_ay).to_numpy())
    lds["fit_rows"] = int(len(lab)); lds["fit_matches"] = ids["train"]
    json.dump(lds, open(os.path.join(a.out, "lds_weights.json"), "w"), indent=1)
    print(f"LDS weights on {len(lab):,} train rows: v max {lds['v']['max']:.2f}, a max {lds['a']['max']:.2f}", flush=True)
    model = LDSModel(tr, params["ModelConfig"], params["OptimizerConfig"], lds=lds)
elif a.variant == "prob":
    model = ProbModel(tr, params["ModelConfig"], params["OptimizerConfig"])
else:
    model = M.SetTransformerModel(tr, params["ModelConfig"], params["OptimizerConfig"])

b = P.collate_phase2([va[i] for i in range(8)])
with torch.no_grad():
    l_var = float(model.step(b)[1])
    out = model.forward(b["features"], b["obs_xy"], b["visible"], b.get("valid", None))
    l_base = float(M.balanced_loss(out[..., :6], b["labels"], b["actor_global_index_lst"], b["visible"], b.get("valid", None))[1])
print(f"variant {a.variant} | train {len(ids['train'])} matches {len(tr)} samples | valid {len(ids['valid'])} matches {len(va)} samples"
      f" | LOSS_W_ACC={M.LOSS_W_ACC} | loss check: variant {l_var:.5f} baseline balanced_loss on the same outputs {l_base:.5f}", flush=True)
assert np.isfinite(l_var) and (abs(l_var - l_base) > 1e-6 or a.variant == "sc")  # sc keeps the baseline loss

if a.smoke:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    opt = model.configure_optimizers()
    for i, bb in enumerate(T._phase2_loader(tr, params, True)):
        bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bb.items()}
        loss = model.step(bb)[1]; opt.zero_grad(); loss.backward(); opt.step()
        print(f"smoke batch {i}: loss {float(loss):.5f}", flush=True)
        if i + 1 >= a.smoke:
            break
    sys.exit(0)

trainer = T._phase2_trainer(argparse.Namespace(devices=1), params)
trainer.fit(model, T._phase2_loader(tr, params, True), T._phase2_loader(va, params, False))
print(trainer.test(model, T._phase2_loader(va, params, False), ckpt_path="best"))  # "test_*" printed here = valid
