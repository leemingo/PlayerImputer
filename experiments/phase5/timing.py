"""Phase 5 timing check: dataset load time, sample counts and GPU step time for one training regime, without a full epoch.

usage: timing.py --regime A|B --data_dir ROOT [--train_batches 600] [--val_batches 200]
A trains on competition 9, B on 9, 10, 135 (valid from the same competitions). Same params as run 001 (imputer_params.json).
Nothing is saved; test is not read.
"""
import os, sys, json, time, argparse, resource

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGIMES = {"A": (9,), "B": (9, 10, 135)}
ap = argparse.ArgumentParser()
ap.add_argument("--regime", choices=REGIMES, required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--train_batches", type=int, default=600)
ap.add_argument("--val_batches", type=int, default=200)
a = ap.parse_args()
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join("/tmp", f"p5timing_{os.getpid()}")
sys.path.insert(0, REPO); os.chdir(REPO)

import pandas as pd, mlflow, pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from imputer import phase2 as P
import imputer.components.model as M
import train as T

mlflow.pytorch.autolog(disable=True)
params = json.load(open(os.path.join(REPO, "imputer_params.json")))["transformer"]
sp = pd.read_csv(os.path.join(a.data_dir, "out", "splits.csv"))
ids = {s: sp[sp.competition.isin(REGIMES[a.regime]) & (sp.split == s)].match_id.tolist() for s in ("train", "valid")}
t0 = time.time()
tr = P.Phase2Windows("train", 5, match_ids=ids["train"], root=a.data_dir)
va = P.Phase2Windows("valid", 5, stats=tr.stats, match_ids=ids["valid"], root=a.data_dir)
t_load = time.time() - t0
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class Clock(Callback):
    def __init__(self): self.t = {}
    def on_train_batch_end(self, trainer, *args):
        if trainer.global_step in (100, a.train_batches): self.t[trainer.global_step] = time.time()
    def on_validation_start(self, trainer, *args): self.t["v0"] = time.time()
    def on_validation_end(self, trainer, *args): self.t["v1"] = time.time()


ck = Clock()
model = M.SetTransformerModel(tr, params["ModelConfig"], params["OptimizerConfig"])
trainer = pl.Trainer(callbacks=[ck], enable_progress_bar=False, logger=False, enable_checkpointing=False, devices=1,
                     accelerator="gpu", max_epochs=1, limit_train_batches=a.train_batches, limit_val_batches=a.val_batches,
                     num_sanity_val_steps=0, gradient_clip_val=params["TrainerConfig"]["gradient_clip_val"])
trainer.fit(model, T._phase2_loader(tr, params, True), T._phase2_loader(va, params, False))
step = (ck.t[a.train_batches] - ck.t[100]) / (a.train_batches - 100)
vstep = (ck.t["v1"] - ck.t["v0"]) / a.val_batches
bs = params["DataConfig"]["batch_size"]
n_tr, n_va = -(-len(tr) // bs), -(-len(va) // bs)
epoch_min = (n_tr * step + n_va * vstep) / 60
print(json.dumps(dict(regime=a.regime, train_matches=len(ids["train"]), valid_matches=len(ids["valid"]), train_samples=len(tr),
                      valid_samples=len(va), load_s=round(t_load, 1), main_rss_gb=round(rss_gb, 2), train_step_s=round(step, 4),
                      val_batch_s=round(vstep, 4), train_steps_per_epoch=n_tr, val_batches=n_va, epoch_min=round(epoch_min, 2),
                      h_100_epochs=round(epoch_min * 100 / 60, 1), h_200_epochs=round(epoch_min * 200 / 60, 1))), flush=True)
