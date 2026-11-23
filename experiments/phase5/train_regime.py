"""Phase 5: train one regime (A: competition 9; B: 9, 10, 135) on the multi-competition dataset.

usage: train_regime.py --regime A|B --data_dir ROOT --out DIR [--w_acc 2.0] [--seed 0]
Run 001 setup otherwise (imputer_params.json transformer block, train.py _phase2_trainer: EarlyStopping patience 5, min_epochs 100,
max_epochs 200, one best_model.ckpt). Normalisation statistics are fitted on the regime's train matches and saved to
DIR/phase2_stats.json. Early stopping uses the valid matches of the trained competitions. Test is not read here; it is evaluated
separately, once per competition, with the saved statistics.
"""
import os, sys, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGIMES = {"A": (9,), "B": (9, 10, 135)}
ap = argparse.ArgumentParser()
ap.add_argument("--regime", choices=REGIMES, required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--w_acc", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns")
sys.path.insert(0, REPO); os.chdir(REPO)

import torch, mlflow, pandas as pd, pytorch_lightning as pl
from imputer import phase2 as P
import imputer.components.model as M
import train as T

mlflow.pytorch.autolog(disable=True)
M.LOSS_W_ACC = a.w_acc
params = json.load(open(os.path.join(REPO, "imputer_params.json")))["transformer"]
params["save_path"] = a.out
sp = pd.read_csv(os.path.join(a.data_dir, "out", "splits.csv"))
ids = {s: sorted(sp[sp.competition.isin(REGIMES[a.regime]) & (sp.split == s)].match_id.astype(int).tolist()) for s in ("train", "valid")}
json.dump(dict(params, regime=a.regime, competitions=REGIMES[a.regime], w_acc=a.w_acc, seed=a.seed, data=a.data_dir, match_ids=ids),
          open(os.path.join(a.out, "params.json"), "w"), indent=1)
pl.seed_everything(a.seed, workers=True)

tr = P.Phase2Windows("train", 5, match_ids=ids["train"], root=a.data_dir)
json.dump(tr.stats, open(os.path.join(a.out, T.PHASE2_STATS_FILE), "w"), indent=1)
va = P.Phase2Windows("valid", 5, stats=tr.stats, match_ids=ids["valid"], root=a.data_dir)
model = M.SetTransformerModel(tr, params["ModelConfig"], params["OptimizerConfig"])

b = P.collate_phase2([va[i] for i in range(4)])
with torch.no_grad():
    l_w = float(model.step(b)[1]); M.LOSS_W_ACC = 0.0; l_0 = float(model.step(b)[1]); M.LOSS_W_ACC = a.w_acc
print(f"regime {a.regime} {REGIMES[a.regime]} | train {len(ids['train'])} matches {len(tr)} samples | valid {len(ids['valid'])} matches {len(va)} samples"
      f" | LOSS_W_ACC={M.LOSS_W_ACC} check w={l_w:.5f} 0={l_0:.5f}", flush=True)
assert (l_w > l_0) == (a.w_acc > 0)

trainer = T._phase2_trainer(argparse.Namespace(devices=1), params)
trainer.fit(model, T._phase2_loader(tr, params, True), T._phase2_loader(va, params, False))
print(trainer.test(model, T._phase2_loader(va, params, False), ckpt_path="best"))
