"""Phase 3 acceleration experiment: fine-tune the run-001 epoch-65 checkpoint with a different acceleration loss weight.

usage: acc_finetune.py --w_acc 0.2 --epochs 15 --out DIR [--smoke]
LOSS_W_ACC is a module constant that _masked_loss reads at call time; it is overridden here (repo code unchanged).
Same data, normalisation stats (run 001), model/optimizer/trainer config as run 001; fresh Adam; fixed epochs (no early stop);
keeps one checkpoint (lowest own val_loss). Valid only; test is not touched.
"""
import os, sys, json, time, argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("PHASE2_DATA", os.path.expanduser("~/scoutgpt_data/_phase2_dataset"))
RUN1 = os.path.join(REPO, "stores/transformer/001")
ap = argparse.ArgumentParser()
ap.add_argument("--w_acc", type=float, required=True)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true", help="CPU, 2 train + 1 valid match, 2 batches")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(a.out, "mlruns")
sys.path.insert(0, REPO); os.chdir(REPO)

import torch, mlflow, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from imputer import phase2 as P
import imputer.components.model as M

mlflow.pytorch.autolog(disable=True)  # run 001's autolog stored an extra 16 MB model copy; keep only best_model.ckpt here
M.LOSS_W_ACC = a.w_acc
params = json.load(open(os.path.join(RUN1, "params.json")))
stats = json.load(open(os.path.join(RUN1, "phase2_stats.json")))
pl.seed_everything(a.seed, workers=True)

ids = {"train": None, "valid": None}
dc, tc = dict(params["DataConfig"]), dict(params["TrainerConfig"])
tc.update(max_epochs=a.epochs, min_epochs=a.epochs)
if a.smoke:
    sp = P.read_splits(DATA)
    ids = {s: [m for m, v in sorted(sp.items()) if v == s][:n] for s, n in (("train", 2), ("valid", 1))}
    dc = {"batch_size": 4, "num_workers": 0, "pin_memory": False}
    tc.update(accelerator="cpu", limit_train_batches=2, limit_val_batches=2)
tr = P.Phase2Windows("train", 5, stats=stats, root=DATA, match_ids=ids["train"])
va = P.Phase2Windows("valid", 5, stats=stats, root=DATA, match_ids=ids["valid"])
ld = lambda ds, sh: DataLoader(ds, shuffle=sh, collate_fn=P.collate_phase2, **dc)

model = M.SetTransformerModel(tr, params["ModelConfig"], params["OptimizerConfig"])
ck = torch.load(os.path.join(RUN1, "best_model.ckpt"), map_location="cpu", weights_only=False)
assert ck["epoch"] == 65, ck["epoch"]
model.load_state_dict(ck["state_dict"])

# the override must reach the loss: same batch, LOSS_W_ACC vs 0 must differ by exactly w_acc * (acc huber term)
b = P.collate_phase2([va[i] for i in range(4)])
with torch.no_grad():
    l_w = float(model.step(b)[1]); M.LOSS_W_ACC = 0.0; l_0 = float(model.step(b)[1]); M.LOSS_W_ACC = a.w_acc
print(f"LOSS_W_ACC={M.LOSS_W_ACC} | check batch loss with w={l_w:.5f}, with 0={l_0:.5f}", flush=True)
assert (l_w > l_0) == (a.w_acc > 0)


class EpochLog(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        print(f"epoch {trainer.current_epoch} {time.strftime('%Y-%m-%d %H:%M:%S')} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items())), flush=True)


ckcb = ModelCheckpoint(monitor="val_loss", dirpath=a.out, filename="best_model", save_top_k=1, mode="min")
trainer = pl.Trainer(callbacks=[ckcb, EpochLog()], enable_progress_bar=False, logger=CSVLogger(a.out, name="csv"),
                     default_root_dir=a.out, devices=1, **tc)
v0 = trainer.validate(model, ld(va, False), verbose=False)[0]
print(f"start (epoch-65 weights, this w_acc): val_loss={v0['val_loss']:.4f} val_euclidean={v0['val_euclidean']:.4f}", flush=True)
t = time.time()
trainer.fit(model, ld(tr, True), ld(va, False))
summ = dict(w_acc=a.w_acc, epochs=a.epochs, seed=a.seed, start_val_loss=v0["val_loss"], best_val_loss=float(ckcb.best_model_score),
            best_path=ckcb.best_model_path, fit_minutes=round((time.time() - t) / 60, 1), smoke=a.smoke)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print("summary", summ, flush=True)
