"""Snapshot figures v2: home blue / away red, prediction in a lighter same-family colour, jersey numbers inside the circles.
Per snapshot two figures: (overlay) truth+prediction on one pitch + table; (side) truth | prediction pitches + home/away tables."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

HERE = os.environ["EVAL_DIR"]  # folder written by eval_valid_ckpt.py (preds_valid.parquet, metrics_valid.csv)
OUT = os.environ.get("FIG_OUT", os.path.expanduser("~/MHL/figures/playerimputer_phase3_epoch65"))
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(os.environ.get("PHASE2_DATA", os.path.expanduser("~/scoutgpt_data/_phase2_dataset")), "out/players")
PICKS = [(4077665, 891), (4077652, 2845), (4077641, 2654), (4077633, 1239)]  # same snapshots as before (seed 65)
GT = {True: "#1f5fbf", False: "#d0312d"}    # home blue, away red (tracking truth)
PR = {True: "#5ab4ee", False: "#f5a04a"}    # prediction: light blue, orange
INK, INK2, MUTED, GRID = "#0b0b0b", "#4a4946", "#9a9994", "#e2e1dc"
MIN_SPD = 0.5
ARROWS = "→↗↑↖←↙↓↘"
plt.rcParams.update({"font.size": 12, "figure.facecolor": "white"})

d = pd.read_parquet(os.path.join(HERE, "preds_valid.parquet"))
d = d[d.has_label]


def pitch(ax, arrow=True):
    kw = dict(color=MUTED, lw=1.0, zorder=0)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, **kw))
    ax.plot([0, 0], [-34, 34], **kw); ax.add_patch(Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * 52.5 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * 52.5 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, **kw))
    if arrow:
        ax.annotate("", (12, -36.0), (-12, -36.0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        ax.text(13.5, -36.0, "event team attacks (0°)", ha="left", va="center", fontsize=11, color=INK2)
    ax.set_xlim(-53.5, 53.5); ax.set_ylim(-37.5, 35); ax.set_aspect("equal"); ax.axis("off")


def direction(vx, vy):
    if np.hypot(vx, vy) < MIN_SPD:
        return None, "  –  "
    a = np.degrees(np.arctan2(vy, vx))
    return a, f"{ARROWS[int(((a % 360) + 22.5) // 45) % 8]}{a:+4.0f}°"


def disc(ax, x, y, col, num, z, size=430, ring=False):
    ax.scatter(x, y, s=size, color=col, edgecolor="white", lw=1.4, zorder=z)
    ax.text(x, y, num, color="white", fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=z + 0.1)
    if ring:
        ax.add_patch(Circle((x, y), 2.9, fill=False, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=z - 0.2))


R_DISC, T_ARROW = 1.6, 2.0  # arrow starts at the circle edge (m) and is T_ARROW seconds of motion long


def arrow(ax, x, y, vx, vy, col, ls, z):
    spd = np.hypot(vx, vy)
    if spd < 1e-6:
        return
    ux, uy = vx / spd, vy / spd
    x0, y0 = x + ux * R_DISC, y + uy * R_DISC
    ax.annotate("", (x0 + vx * T_ARROW, y0 + vy * T_ARROW), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=16), zorder=z)


def rows_for(s):
    cells, home = [], []
    for _, r in s.iterrows():
        ta, ts = direction(r.lbl_vx, r.lbl_vy); pa, ps = direction(r.p_vx, r.p_vy)
        dd = "–" if ta is None or pa is None else f"{abs((pa - ta + 180) % 360 - 180):.0f}°"
        cells.append([str(r.jersey), "vis" if r.observed else "HID",
                      f"{np.hypot(r.lbl_vx, r.lbl_vy):3.1f} {np.hypot(r.p_vx, r.p_vy):3.1f}", f"{ts} {ps}", dd,
                      f"{np.hypot(r.lbl_ax, r.lbl_ay):3.1f} {np.hypot(r.p_ax, r.p_ay):3.1f}",
                      f"{np.hypot(r.p_x - r.lbl_x, r.p_y - r.lbl_y):.1f}"])
        home.append(bool(r.is_home))
    return cells, home


HDR = ["#", "360", "speed m/s\ntrue pred", "direction\ntrue    pred", "Δdir", "|acc| m/s²\ntrue pred", "pos\nerr m"]
CW = [0.07, 0.08, 0.16, 0.30, 0.09, 0.17, 0.10]


def table(ax, cells, home, fs, rh, hh):
    tb = ax.table(cellText=cells, colLabels=HDR, loc="upper center", cellLoc="center", colWidths=CW)
    tb.auto_set_font_size(False); tb.set_fontsize(fs)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID); c.set_linewidth(0.6); c.get_text().set_fontfamily("DejaVu Sans Mono")
        if i == 0:
            c.set_height(hh); c.get_text().set_fontfamily("DejaVu Sans"); c.get_text().set_color(INK2); c.set_facecolor("#f4f3f0")
        else:
            c.set_height(rh)
            if j == 0:
                c.set_facecolor(GT[home[i - 1]]); c.get_text().set_color("white"); c.get_text().set_fontweight("bold")
            elif cells[i - 1][1] == "HID":
                c.set_facecolor("#f3f3f1")


def legend_handles(both):
    h = [Line2D([], [], marker="o", ls="", color=GT[True], markersize=12, label="home · tracking truth"),
         Line2D([], [], marker="o", ls="", color=GT[False], markersize=12, label="away · tracking truth")]
    if both:
        h += [Line2D([], [], marker="D", ls="", color=PR[True], markersize=10, label="home · prediction"),
              Line2D([], [], marker="D", ls="", color=PR[False], markersize=10, label="away · prediction")]
    else:
        h += [Line2D([], [], marker="o", ls="", color=PR[True], markersize=12, label="home · prediction"),
              Line2D([], [], marker="o", ls="", color=PR[False], markersize=12, label="away · prediction")]
    h += [Line2D([], [], color=INK2, lw=2.2, label="velocity truth (2 s, from circle edge)"),
          Line2D([], [], color=INK2, lw=2.2, ls="--", label="velocity prediction (2 s)"),
          Line2D([], [], marker="x", ls="", color=INK2, markersize=8, label="360 observation"),
          Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=15, label="hidden in 360 (dashed ring)"),
          Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=15, label="event actor")]
    return h


for k, (m, sidx) in enumerate(PICKS):
    jer = pd.read_parquet(os.path.join(DATA, f"{m}.parquet"), columns=["snap_idx", "object_id", "jersey"])
    s = d[(d.match_id == m) & (d.snap_idx == sidx)].merge(jer[jer.snap_idx == sidx], on=["snap_idx", "object_id"], how="left")
    s = s.sort_values(["is_home", "jersey"], ascending=[False, True]).reset_index(drop=True)
    vis, hid = s[s.observed], s[~s.observed]
    ev = lambda q: np.hypot(q.p_vx - q.lbl_vx, q.p_vy - q.lbl_vy).median()
    head = (f"match {m} · period {int(s.period.iloc[0])} · t = {s.t_s.iloc[0]:.1f} s · visible {len(vis)} / hidden {len(hid)} · "
            f"epoch-65 transformer vs DFL tracking (valid)")
    sub = (f"median errors: hidden position {np.hypot(hid.p_x - hid.lbl_x, hid.p_y - hid.lbl_y).median():.2f} m · "
           f"visible velocity {ev(vis):.2f} m/s · hidden velocity {ev(hid):.2f} m/s · "
           f"direction 0° = event team's attack (+x), CCW +, '–' if speed < {MIN_SPD} m/s · grey rows = hidden in 360")
    cells, home = rows_for(s)

    # ---------- overlay ----------
    fig = plt.figure(figsize=(19, 8.4))
    ax = fig.add_axes([0.0, 0.125, 0.545, 0.78]); pitch(ax)
    for _, r in s.iterrows():
        h = bool(r.is_home)
        ax.plot([r.lbl_x, r.p_x], [r.lbl_y, r.p_y], color=INK2, lw=1.0, ls=":", zorder=1)
        arrow(ax, r.lbl_x, r.lbl_y, r.lbl_vx, r.lbl_vy, GT[h], "-", 3)
        arrow(ax, r.p_x, r.p_y, r.p_vx, r.p_vy, PR[h], "--", 3)
        if r.observed:
            ax.scatter(r.obs_x, r.obs_y, s=40, marker="x", color=INK2, lw=1.1, alpha=0.7, zorder=2)
        ax.scatter(r.p_x, r.p_y, s=150, marker="D", color=PR[h], edgecolor="white", lw=1.2, zorder=5)
        disc(ax, r.lbl_x, r.lbl_y, GT[h], str(r.jersey), 6, ring=not r.observed)
        if r.actor:
            ax.scatter(r.lbl_x, r.lbl_y, s=900, facecolor="none", edgecolor=INK, lw=1.5, marker="s", zorder=4)
    fig.legend(handles=legend_handles(True), loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=4, frameon=False, fontsize=11,
               columnspacing=1.2, handletextpad=0.4)
    tax = fig.add_axes([0.55, 0.01, 0.445, 0.86]); tax.axis("off")
    table(tax, cells, home, 11.5, 0.039, 0.07)
    fig.text(0.006, 0.965, head, fontsize=15, color=INK)
    fig.text(0.006, 0.925, sub, fontsize=11, color=INK2)
    fig.savefig(os.path.join(OUT, f"fig1{'abcd'[k]}_overlay_{m}_{sidx}.png"), dpi=140); plt.close(fig)

    # ---------- side by side ----------
    fig = plt.figure(figsize=(19, 13.2))
    for col, (title, pred) in enumerate((("DFL tracking truth", False), ("model prediction (epoch 65)", True))):
        ax = fig.add_axes([0.005 + col * 0.5, 0.47, 0.49, 0.45]); pitch(ax)
        ax.text(-52.5, 35.6, title, fontsize=15, color=INK, va="bottom")
        for _, r in s.iterrows():
            h = bool(r.is_home)
            if r.observed:
                ax.scatter(r.obs_x, r.obs_y, s=40, marker="x", color=INK2, lw=1.1, alpha=0.7, zorder=2)
            if pred:
                arrow(ax, r.p_x, r.p_y, r.p_vx, r.p_vy, PR[h], "--", 3)
                disc(ax, r.p_x, r.p_y, PR[h], str(r.jersey), 6, ring=not r.observed)
            else:
                arrow(ax, r.lbl_x, r.lbl_y, r.lbl_vx, r.lbl_vy, GT[h], "-", 3)
                disc(ax, r.lbl_x, r.lbl_y, GT[h], str(r.jersey), 6, ring=not r.observed)
                if r.actor:
                    ax.scatter(r.lbl_x, r.lbl_y, s=900, facecolor="none", edgecolor=INK, lw=1.5, marker="s", zorder=4)
    fig.legend(handles=legend_handles(False), loc="lower left", bbox_to_anchor=(0.005, 0.385), ncol=5, frameon=False, fontsize=11,
               columnspacing=1.2, handletextpad=0.4)
    for col, hv in enumerate((True, False)):
        idx = [i for i, h in enumerate(home) if h == hv]
        tax = fig.add_axes([0.01 + col * 0.5, 0.005, 0.475, 0.34]); tax.axis("off")
        tax.text(0.0, 1.0, "home" if hv else "away", transform=tax.transAxes, fontsize=13, color=GT[hv], fontweight="bold", va="bottom")
        table(tax, [cells[i] for i in idx], [home[i] for i in idx], 11.5, 0.072, 0.13)
    fig.text(0.006, 0.975, head, fontsize=15, color=INK)
    fig.text(0.006, 0.952, sub, fontsize=11, color=INK2)
    fig.savefig(os.path.join(OUT, f"fig1{'abcd'[k]}_side_{m}_{sidx}.png"), dpi=140); plt.close(fig)
    print(m, sidx, "jersey missing:", int(s.jersey.isna().sum()))
