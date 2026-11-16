"""Side-by-side table of key_valid.json for ref_epoch65 and the three arms (valid split)."""
import json, os
B = os.environ.get("PHASE3_ACC", os.path.expanduser("~/scoutgpt_data/_phase3_acc"))
runs = {"ref65": f"{B}/ref_epoch65/key_valid.json", **{f"w{w}": f"{B}/w{w}/eval/key_valid.json" for w in ("0.05", "0.2", "0.5", "1.0", "2.0", "5.0") if os.path.exists(f"{B}/w{w}/eval/key_valid.json")}}
K = {r: json.load(open(p)) for r, p in runs.items()}
print("epoch:", {r: k["epoch"] for r, k in K.items()})
rows = ["pos_med", "pos_mean", "vel_vec_rmse", "spd_rmse", "r2", "head_med_deg", "acc_vec_rmse", "acc_zero_rmse", "acc_gain_vs_zero_pct", "acc_corr",
        "acc_true_mag_mean", "acc_pred_mag_mean", "acc_big_true_mean", "acc_big_pred_mean", "spd_hi_true_mean", "spd_hi_pred_mean"]
for t in ("visible", "hidden"):
    print(f"\n[{t}] n={K['ref65'][t]['n']:,}")
    print(f"{'metric':22s}" + "".join(f"{r:>10s}" for r in K))
    for m in rows:
        print(f"{m:22s}" + "".join(f"{K[r][t][m]:10.4f}" for r in K))
