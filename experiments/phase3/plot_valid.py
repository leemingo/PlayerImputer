"""Figures for the epoch-65 valid evaluation (reads preds_valid.parquet + metrics csv written by eval_valid.py)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.lines import Line2D

HERE = os.environ["EVAL_DIR"]  # folder written by eval_valid_ckpt.py (preds_valid.parquet, metrics_valid.csv)
OUT = os.environ.get("FIG_OUT", os.path.expanduser("~/MHL/figures/playerimputer_phase3_epoch65"))
os.makedirs(OUT, exist_ok=True)
B = os.environ.get("PHASE2_BASELINE_TABLES", os.path.expanduser("~/scoutgpt_data/_phase2_baseline/tables"))
TEAM, OPP = "#2a78d6", "#eb6834"          # categorical slots 1, 2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white"})

d = pd.read_parquet(os.path.join(HERE, "preds_valid.parquet"))
d = d[d.has_label].copy()
d["err"] = np.hypot(d.p_x - d.lbl_x, d.p_y - d.lbl_y)


def pitch(ax):
    kw = dict(color=MUTED, lw=0.8, zorder=0)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, **kw))
    ax.plot([0, 0], [-34, 34], **kw); ax.add_patch(Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * 52.5 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * 52.5 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, **kw))
    ax.set_xlim(-56, 56); ax.set_ylim(-37, 37); ax.set_aspect("equal"); ax.axis("off")


# ---------- Figure 1: four random snapshots (one per match, >=4 hidden labelled players) ----------
rng = np.random.default_rng(65)
g = d.groupby(["match_id", "snap_idx"])
cnt = g.observed.agg(n_vis="sum", n=("size")) if False else g.agg(n_vis=("observed", "sum"), n=("observed", "size"))
cnt["n_hid"] = cnt.n - cnt.n_vis
elig = cnt[cnt.n_hid >= 4].reset_index()
picks = []
for m in rng.permutation(elig.match_id.unique())[:4]:
    e = elig[elig.match_id == m]
    picks.append(e.iloc[rng.integers(len(e))][["match_id", "snap_idx"]].tolist())

fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
for ax, (m, sidx) in zip(axs.ravel(), picks):
    s = d[(d.match_id == m) & (d.snap_idx == sidx)]
    pitch(ax)
    for _, r in s.iterrows():
        c = TEAM if r.teammate else OPP
        ax.plot([r.lbl_x, r.p_x], [r.lbl_y, r.p_y], color=INK2, lw=0.7, ls=":", zorder=1)  # position error
        ax.annotate("", (r.lbl_x + r.lbl_vx, r.lbl_y + r.lbl_vy), (r.lbl_x, r.lbl_y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", (r.p_x + r.p_vx, r.p_y + r.p_vy), (r.p_x, r.p_y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, ls="--", shrinkA=0, shrinkB=0), zorder=3)
        ax.scatter(r.lbl_x, r.lbl_y, s=70, color=c, edgecolor="white", lw=1.2, zorder=4)
        ax.scatter(r.p_x, r.p_y, s=60, marker="D", facecolor="white", edgecolor=c, lw=1.6, zorder=5)
        if r.observed:
            ax.scatter(r.obs_x, r.obs_y, s=28, marker="x", color=INK, lw=1.0, zorder=6)
        else:
            ax.add_patch(Circle((r.lbl_x, r.lbl_y), 2.6, fill=False, color=INK, lw=1.0, ls="-", zorder=2))
        if r.actor:
            ax.scatter(r.lbl_x, r.lbl_y, s=230, facecolor="none", edgecolor=INK, lw=1.4, marker="s", zorder=2)
    vis, hid = s[s.observed], s[~s.observed]
    ax.set_title(f"match {m}  period {int(s.period.iloc[0])}  t={s.t_s.iloc[0]:.1f}s   visible {len(vis)} / hidden {len(hid)}\n"
                 f"hidden pos err median {hid.err.median():.2f} m   visible vel err median "
                 f"{np.hypot(vis.p_vx - vis.lbl_vx, vis.p_vy - vis.lbl_vy).median():.2f} m/s", fontsize=9, loc="left")
leg = [Line2D([], [], marker="o", ls="", color=INK2, markersize=8, label="tracking truth (position)"),
       Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK2, markersize=7, label="model prediction (position)"),
       Line2D([], [], color=INK2, lw=1.6, label="truth velocity (arrow = 1 s of motion)"),
       Line2D([], [], color=INK2, lw=1.2, ls="--", label="predicted velocity"),
       Line2D([], [], marker="x", ls="", color=INK, markersize=7, label="StatsBomb 360 observation (visible)"),
       Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=12, label="hidden in 360 (ring)"),
       Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=12, label="event actor"),
       Line2D([], [], marker="o", ls="", color=TEAM, markersize=8, label="event team (attacks +x)"),
       Line2D([], [], marker="o", ls="", color=OPP, markersize=8, label="opponent")]
fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
fig.suptitle("Epoch-65 transformer vs DFL tracking truth, 4 random valid snapshots (seed 65, >=4 hidden players)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(os.path.join(OUT, "fig1_snapshots.png"), dpi=150); plt.close(fig)
print("picks", picks)

# ---------- Figure 2: error by gap bucket, baseline T2_hgb vs epoch 65 (two panels, separate axes) ----------
m = pd.read_csv(os.path.join(HERE, "metrics_valid.csv"))
ORDER = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-5", "5-7.5", "7.5-10", "10-15", "15-30", ">30"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, (task, metric, kind, ylab, ttl) in zip(axs, (
        ("visible", "vel_vec_rmse", "span", "velocity vector RMSE (m/s)", "Visible players: velocity error by sighting span (prev_dt + next_dt, s)"),
        ("hidden", "pos_med", "prev_dt", "median position error (m)", "Hidden players: position error by time since last sighting (prev_dt, s)"))):
    b = pd.read_csv(f"{B}/{task}_valid_all.csv"); b = b[(b.model == "T2_hgb") & (b.group_kind == kind)].set_index("group")[metric]
    t = m[(m.task == task) & (m["var"] == "all") & (m.group_kind == kind)].set_index("group")[metric]
    x = np.arange(len(ORDER))
    for ser, col, lab, mk in ((b, MUTED, "baseline T2_hgb", "o"), (t, TEAM, "transformer epoch 65", "D")):
        y = ser.reindex(ORDER).values
        ax.plot(x, y, color=col, lw=2, marker=mk, markersize=6, markeredgecolor="white", label=lab)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, ORDER, rotation=35, ha="right"); ax.set_ylabel(ylab); ax.set_title(ttl, loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlim(-0.5, len(ORDER) + 1.6); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
fig.suptitle("Valid split, same rows as the Phase 2 baseline (visible 405,625 / hidden 144,969)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig2_error_by_gap.png"), dpi=150); plt.close(fig)

# ---------- Figure 3: predicted vs true, density (vx and hidden x) ----------
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
vis, hid = d[d.observed], d[~d.observed]
for ax, (tv, pv, lim, ttl, lab) in zip(axs, (
        (vis.lbl_vx, vis.p_vx, 9, "Visible players: vx", "m/s"),
        (hid.lbl_vx, hid.p_vx, 9, "Hidden players: vx", "m/s"),
        (hid.lbl_x, hid.p_x, 55, "Hidden players: x position", "m"))):
    hb = ax.hexbin(tv, pv, gridsize=70, extent=(-lim, lim, -lim, lim), bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"tracking truth ({lab})"); ax.set_ylabel(f"prediction ({lab})")
    r = np.corrcoef(tv, pv)[0, 1]
    ax.set_title(f"{ttl}   (n={len(tv):,}, r={r:.3f})", loc="left", fontsize=10)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="rows (log)")
fig.suptitle("Prediction vs tracking truth, valid split, epoch 65 (dashed = perfect prediction)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig3_pred_vs_truth.png"), dpi=150); plt.close(fig)
print("done")
