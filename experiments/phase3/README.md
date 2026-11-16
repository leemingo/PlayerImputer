# Phase 3 experiment scripts

Scripts for the Phase 3 transformer runs on the Phase 2 dataset (`data_build/phase2/SCHEMA.md`). Every script uses the valid split only. The test split is evaluated once, by `train.py` at the end of training.

| script | what it does |
|---|---|
| `acc_finetune.py` | Fine-tunes the run-001 checkpoint (`stores/transformer/001/best_model.ckpt`, epoch 65) for a fixed number of epochs with a different `LOSS_W_ACC`. It reuses run 001's params and normalisation stats (`phase2_stats.json`), uses a fresh Adam optimiser, and keeps one checkpoint, the one with the lowest val_loss. The weight is overridden at run time, so the repo code is unchanged. Before training it checks that the override changes the loss. `--smoke` runs on the CPU with 2 train matches and 1 valid match. |
| `eval_valid_ckpt.py` | Evaluates one checkpoint on valid with the Phase 2b `metric_frame`. It scores the same rows as the baseline (`has_label`, visible = `observed`) and adds acceleration and speed summaries. It writes `metrics_valid.csv`, `key_valid.json` and `preds_valid.parquet`. |
| `compare_acc.py` | Prints `key_valid.json` side by side for the epoch-65 reference and each `LOSS_W_ACC` arm. |
| `plot_valid.py` | Draws the valid figures from the `eval_valid_ckpt.py` output: example snapshots, error by observation gap, and prediction vs truth. |
| `plot_snapshot.py` | Draws one figure per snapshot: truth vs prediction on the pitch, plus a per-player table of speed, direction and acceleration (overlay and side-by-side versions). |

Paths come from environment variables. The defaults are the layout of the original server.

| variable | default | used by |
|---|---|---|
| `PHASE2_DATA` | `~/scoutgpt_data/_phase2_dataset` | all except `compare_acc.py` |
| `PHASE2_BASELINE_SRC` | `~/scoutgpt_data/_phase2_baseline/src` (Phase 2b `common.py`, from `phase2_baseline_report.tgz`, not in this repo) | `eval_valid_ckpt.py` |
| `PHASE2_BASELINE_TABLES` | `~/scoutgpt_data/_phase2_baseline/tables` | `plot_valid.py` |
| `PHASE3_ACC` | `~/scoutgpt_data/_phase3_acc` (one `w<weight>/eval/` folder per arm, plus `ref_epoch65/`) | `compare_acc.py` |
| `EVAL_DIR` | required | plot scripts (an `eval_valid_ckpt.py` output folder) |
| `FIG_OUT` | `~/MHL/figures/playerimputer_phase3_epoch65` | plot scripts |

```bash
# fine-tune with LOSS_W_ACC = 0.5 for 15 epochs, then evaluate on valid
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/phase3/acc_finetune.py --w_acc 0.5 --epochs 15 --out ~/scoutgpt_data/_phase3_acc/w0.5
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/phase3/eval_valid_ckpt.py --ckpt ~/scoutgpt_data/_phase3_acc/w0.5/best_model.ckpt --out ~/scoutgpt_data/_phase3_acc/w0.5/eval --tag w0.5
.venv/bin/python experiments/phase3/compare_acc.py
EVAL_DIR=~/scoutgpt_data/_phase3_acc/w0.5/eval .venv/bin/python experiments/phase3/plot_snapshot.py
```

Check done before committing: `eval_valid_ckpt.py` on the run-001 checkpoint reproduces the earlier `metrics_valid.csv` with a max abs diff of 0.0. The plot scripts reproduce the saved figures byte for byte. `acc_finetune.py --smoke` runs.
