"""Phase 6 loss / output variants of SetTransformerModel. imputer/ is not modified: everything here subclasses or reuses it.

ProbModel (T1): adds a log-variance head next to the mean heads. Output (B, W, N, 12) = 6 means (vx, vy, x, y, ax, ay, same units
    as the labels) + 6 log-variances of the residual in TARGET_SCALE units. Loss = beta-NLL (Seitzer et al. 2022, beta 0.5, the
    variance factor var^beta is detached), grouped and weighted exactly like balanced_loss (position / velocity / acceleration groups
    with LOSS_W_POS / LOSS_W_VEL / LOSS_W_ACC, visible and hidden subsets averaged separately) plus the same cosine heading term on the
    means. The means stay the point estimate; metrics and val_euclidean use the means.
LDSModel (T2): balanced_loss with Label Distribution Smoothing weights (Yang et al. 2021) on the velocity and acceleration Huber terms,
    row-wise from the true |v| and |a|; position and heading terms unchanged. Weight tables are fitted on train rows (fit_lds_weights).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

import imputer.components.model as M
from imputer.components.model import SetTransformerModel, balanced_loss

LOGVAR_LO, LOGVAR_HI = -7.0, 5.0  # soft range of the log-variance in TARGET_SCALE units (std 0.03 .. 12 scale units)
BETA = 0.5
COMPS6 = ("vx", "vy", "x", "y", "ax", "ay")


def _center(preds, targets, visible, valid):
    """Middle timestep, labelled and valid tokens: returns preds (T, D'), targets (T, 6), visible (T,)."""
    W = targets.shape[1]
    p, t, v = preds[:, W // 2], targets[:, W // 2], visible[:, W // 2].bool()
    m = ~torch.isnan(t).any(dim=-1)
    if valid is not None:
        m = m & valid[:, W // 2].bool()
    return p[m], t[m], v[m]


def _balanced(fn, p, t, v):
    """visible / hidden subsets averaged separately with LOSS_W_VISIBLE / LOSS_W_HIDDEN, as balanced_loss."""
    total, ws = p.sum() * 0.0, 0.0
    for sub, w in ((v, M.LOSS_W_VISIBLE), (~v, M.LOSS_W_HIDDEN)):
        if sub.any():
            total = total + w * fn(p[sub], t[sub]); ws += w
    return total / ws if ws > 0 else total


def _heading(mean, t):
    moving = t[:, :2].norm(dim=1) > M.HEADING_MIN_SPEED
    if moving.any():
        return M.LOSS_W_COS * (1 - F.cosine_similarity(mean[moving, :2], t[moving, :2], dim=1)).mean()
    return mean.sum() * 0.0


# ---------------------------------------------------------------- T1
def beta_nll_terms(mean, logvar, t):
    scale = torch.tensor([M.TARGET_SCALE[c] for c in COMPS6], device=mean.device, dtype=mean.dtype)
    z2 = ((mean - t) / scale) ** 2
    nll = 0.5 * (logvar + z2 * torch.exp(-logvar))
    return nll * torch.exp(logvar).detach() ** BETA  # (T, 6)


def _prob_subset_loss(p, t):
    mean, logvar = p[:, :6], p[:, 6:]
    term = beta_nll_terms(mean, logvar, t)
    loss = M.LOSS_W_POS * term[:, 2:4].mean() + M.LOSS_W_VEL * term[:, 0:2].mean() + M.LOSS_W_ACC * term[:, 4:6].mean()
    return loss + _heading(mean, t)


class ProbModel(SetTransformerModel):
    def __init__(self, train_dataset, model_config=None, optimizer_params=None):
        super().__init__(train_dataset, model_config, optimizer_params)
        h = self.model.fc_xy[0].in_features
        self.model.fc_logvar = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 6))
        self._hidden = None
        self.model.fc_xy.register_forward_pre_hook(lambda mod, inp: setattr(self, "_hidden", inp[0]))

    def forward(self, x, obs_xy, visible, valid=None):
        mean = self.model(x, obs_xy, visible, valid)
        raw = self.model.fc_logvar(self._hidden)
        logvar = LOGVAR_LO + (LOGVAR_HI - LOGVAR_LO) * torch.sigmoid(raw)
        return torch.cat([mean, logvar], dim=-1)

    def step(self, batch):
        out = self.forward(batch["features"], batch["obs_xy"], batch["visible"], batch.get("valid", None))
        mean = out[..., :6]
        self._last_outputs = mean
        mae, _, euc, cos = balanced_loss(mean, batch["labels"], batch["actor_global_index_lst"], batch["visible"], batch.get("valid", None))
        p, t, v = _center(out, batch["labels"], batch["visible"], batch.get("valid", None))
        return mae, _balanced(_prob_subset_loss, p, t, v), euc, cos


# ---------------------------------------------------------------- T2
def fit_lds_weights(v_true, a_true, bin_w=0.25, sigma_bins=2.0, power=0.5, clip=10.0, v_max=13.0, a_max=8.0):
    """v_true, a_true: magnitudes of all labelled train rows. Returns a json-able dict of bin edges and weights
    (weight = smoothed density^-power, normalised to mean 1 over the train rows, clipped, renormalised)."""
    out = {"bin_w": bin_w, "sigma_bins": sigma_bins, "power": power, "clip": clip}
    for name, x, hi in (("v", v_true, v_max), ("a", a_true, a_max)):
        edges = np.arange(0.0, hi + bin_w, bin_w)
        idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 1)
        cnt = np.bincount(idx, minlength=len(edges)).astype(float)
        dens = gaussian_filter1d(cnt, sigma_bins, mode="nearest") / len(x)
        w = np.where(dens > 0, dens, dens[dens > 0].min()) ** (-power)
        w = w / (w[idx].mean())
        w = np.minimum(w, clip); w = w / w[idx].mean()
        out[name] = {"edges": edges.tolist(), "w": w.tolist(), "row_mean_after": float(w[idx].mean()), "max": float(w.max())}
    return out


def _lookup(x, edges, w):
    idx = torch.clamp(torch.bucketize(x, edges, right=True) - 1, 0, len(edges) - 1)
    return w[idx]


class LDSModel(SetTransformerModel):
    def __init__(self, train_dataset, model_config=None, optimizer_params=None, lds=None):
        super().__init__(train_dataset, model_config, optimizer_params)
        if lds is None:
            raise ValueError("LDSModel needs the fitted weight tables (lds, from <run>/lds_weights.json)")
        for k in ("v", "a"):
            self.register_buffer(f"lds_{k}_edges", torch.tensor(lds[k]["edges"], dtype=torch.float32))
            self.register_buffer(f"lds_{k}_w", torch.tensor(lds[k]["w"], dtype=torch.float32))

    def _subset_loss(self, p, t):
        scale = torch.tensor([M.TARGET_SCALE[c] for c in COMPS6], device=p.device, dtype=p.dtype)
        huber = F.smooth_l1_loss(p / scale, t / scale, beta=M.HUBER_BETA, reduction="none")
        wv = _lookup(t[:, 0:2].norm(dim=1), self.lds_v_edges, self.lds_v_w)
        wa = _lookup(t[:, 4:6].norm(dim=1), self.lds_a_edges, self.lds_a_w)
        loss = M.LOSS_W_POS * huber[:, 2:4].mean() + M.LOSS_W_VEL * (wv * huber[:, 0:2].mean(dim=1)).mean()
        loss = loss + M.LOSS_W_ACC * (wa * huber[:, 4:6].mean(dim=1)).mean()
        return loss + _heading(p, t)

    def step(self, batch):
        out = self.forward(batch["features"], batch["obs_xy"], batch["visible"], batch.get("valid", None))
        self._last_outputs = out
        mae, _, euc, cos = balanced_loss(out, batch["labels"], batch["actor_global_index_lst"], batch["visible"], batch.get("valid", None))
        p, t, v = _center(out, batch["labels"], batch["visible"], batch.get("valid", None))
        return mae, _balanced(self._subset_loss, p, t, v), euc, cos
