"""Phase 3 acceleration experiment: train from scratch with a different acceleration loss weight, run 001 setup otherwise.

usage: acc_scratch.py --w_acc 1.0 --out DIR [--seed 0]
Same data, transformer params (imputer_params.json), trainer and callbacks as run 001 (train.py _phase2_trainer: EarlyStopping
patience 5, min_epochs 100, max_epochs 200, one best_model.ckpt). LOSS_W_ACC is overridden as in acc_finetune.py.
train_phase2 is not called because it also evaluates the test split; this script evaluates valid only.
"""
import os, sys, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("PHASE2_DATA", os.path.expanduser("~/scoutgpt_data/_phase2_dataset"))
ap = argparse.ArgumentParser()
ap.add_argument("--w_acc", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns")
sys.path.insert(0, REPO); os.chdir(REPO)

import torch, mlflow, pytorch_lightning as pl
from imputer import phase2 as P
import imputer.components.model as M
import train as T

mlflow.pytorch.autolog(disable=True)  # keep only best_model.ckpt
M.LOSS_W_ACC = a.w_acc
params = json.load(open(os.path.join(REPO, "imputer_params.json")))["transformer"]
params["save_path"] = a.out
json.dump(dict(params, w_acc=a.w_acc, seed=a.seed, data=DATA), open(os.path.join(a.out, "params.json"), "w"), indent=1)
pl.seed_everything(a.seed, workers=True)

tr = P.Phase2Windows("train", 5, root=DATA)
stats = tr.stats
json.dump(stats, open(os.path.join(a.out, T.PHASE2_STATS_FILE), "w"), indent=1)
ref = json.load(open(os.path.join(REPO, "stores/transformer/001", T.PHASE2_STATS_FILE)))
assert json.loads(json.dumps(stats)) == ref, "train statistics differ from run 001"
va = P.Phase2Windows("valid", 5, stats=stats, root=DATA)
model = M.SetTransformerModel(tr, params["ModelConfig"], params["OptimizerConfig"])

# the override must reach the loss: same batch, LOSS_W_ACC vs 0 must differ
b = P.collate_phase2([va[i] for i in range(4)])
with torch.no_grad():
    l_w = float(model.step(b)[1]); M.LOSS_W_ACC = 0.0; l_0 = float(model.step(b)[1]); M.LOSS_W_ACC = a.w_acc
print(f"LOSS_W_ACC={M.LOSS_W_ACC} | check batch loss with w={l_w:.5f}, with 0={l_0:.5f} | train {len(tr)}, valid {len(va)}", flush=True)
assert (l_w > l_0) == (a.w_acc > 0)

trainer = T._phase2_trainer(argparse.Namespace(devices=1), params)
trainer.fit(model, T._phase2_loader(tr, params, True), T._phase2_loader(va, params, False))
print(trainer.test(model, T._phase2_loader(va, params, False), ckpt_path="best"))
