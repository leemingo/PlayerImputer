"""CPU wiring test for `train.py --dataset phase2`: train_phase2 / test_phase2 on a few matches, 2 batches, plus the checkpoint load path.

Run from the repo root:  .venv/bin/python -B tests/smoke_train_phase2.py
Writes only to a temporary directory (TMPDIR). Reads 2 train + 1 valid + 1 test match from the Phase 2 parquet files (read-only).
"""
import os
import sys
import json
import time
import copy
import tempfile
import argparse
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TMP = tempfile.mkdtemp(prefix="p2wiring_")
os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.join(TMP, "mlruns")  # model.py calls mlflow autolog on import

import torch
import train as T  # noqa: E402
from imputer import phase2 as P
from imputer.components import press

torch.manual_seed(0)
torch.set_num_threads(4)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        check.failed += 1
check.failed = 0


t0 = time.time()
splits = P.read_splits()
ids = {s: [m for m, v in sorted(splits.items()) if v == s] for s in ("train", "valid", "test")}
match_ids = {"train": ids["train"][:2], "valid": ids["valid"][:1], "test": ids["test"][:1]}

params = json.load(open("imputer_params.json"))["transformer"]
params["ModelConfig"]["dropout"] = 0.0
params["DataConfig"] = {"batch_size": 4, "num_workers": 0, "pin_memory": False}
params["TrainerConfig"].update({"accelerator": "cpu", "max_epochs": 1, "min_epochs": 1, "limit_train_batches": 2,
                                "limit_val_batches": 2, "limit_test_batches": 2})
params["TrainerConfig"]["default_root_dir"] = TMP  # lightning_logs go to the temporary directory
params["save_path"] = os.path.join(TMP, "run")
os.makedirs(params["save_path"])
args = argparse.Namespace(model="transformer", trial=1, window=5, data_dir=None, devices=1, dataset="phase2")

print("== train_phase2 (CPU, 2 train batches of 4)")
model = T.train_phase2(args, params, match_ids=match_ids)
stats_path = os.path.join(params["save_path"], T.PHASE2_STATS_FILE)
ckpt_path = os.path.join(params["save_path"], "best_model.ckpt")
check("best_model.ckpt written", os.path.exists(ckpt_path))
check("phase2_stats.json written next to the run output", os.path.exists(stats_path))
saved = json.load(open(stats_path))
check("stats round trip is exact (json -> dict == fitted stats), window included", saved == P.Phase2Windows("train", 5, match_ids=match_ids["train"]).stats and saved["window"] == 5)
check("stats were fitted on the train matches only", set(saved["fit_matches"]) == set(match_ids["train"]))
check("model was built with dim_output 6", model.model.dim_output == 6 and hasattr(model.model, "fc_a"))

print("== test_phase2 reuses the saved statistics")
orig_ds = P.Phase2Windows
seen = {}
class Spy(orig_ds):
    def __init__(self, *a, **k):
        seen["stats"] = k.get("stats")
        k["match_ids"] = match_ids["test"]
        super().__init__(*a, **k)
T.phase2.Phase2Windows = Spy
try:
    T.test_phase2(args, params)
    ok = True
except Exception as e:
    print("   ", type(e).__name__, e)
    ok = False
check("test_phase2 loads best_model.ckpt through trainer.test(ckpt_path=...) and evaluates", ok)
check("test_phase2 passed the saved (not refitted) stats", seen.get("stats") == saved)
try:
    T.test_phase2(argparse.Namespace(**{**vars(args), "window": 3}), params)
    ok = False
except ValueError as e:
    ok = "window" in str(e)
T.phase2.Phase2Windows = orig_ds
check("a different --window than the saved one is refused", ok)

print("== checkpoint load path imputer/components/base.py:58 (torch.load) on torch", torch.__version__)
comp = press.SetTransformerComponent.__new__(press.SetTransformerComponent)  # skip __init__ (it builds a Trainer with devices=[3])
comp.model = copy.deepcopy(model)
for p_ in comp.model.parameters():
    p_.data.zero_()
try:
    comp.load(Path(ckpt_path))
    loaded = True
except Exception as e:
    print("   ", type(e).__name__, str(e)[:300])
    loaded = False
check("exPressComponent.load() reads a Lightning checkpoint on this torch", loaded)
if loaded:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check("loaded weights equal the checkpoint's state_dict", all(torch.equal(comp.model.state_dict()[k], v) for k, v in ck["state_dict"].items()))

print(f"done in {time.time() - t0:.1f}s, failures: {check.failed}")
sys.exit(1 if check.failed else 0)
