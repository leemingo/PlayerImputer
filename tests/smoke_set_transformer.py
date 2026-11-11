"""CPU smoke test for the identity-based SetTransformer and its loss on synthetic tensors (no dataset needed).

Run from the repo root:  .venv/bin/python -B tests/smoke_set_transformer.py
"""
import os
import sys
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

import imputer.config as config
from imputer.components import model as M

torch.manual_seed(0)
torch.set_num_threads(4)

# README 학습 명령과 같은 23개 xfns. categorical_indices는 ImputerDataset과 같은 방식으로 계산함.
XFNS = ["prevAgentX", "prevAgentY", "prevAgentTime", "nextAgentX", "nextAgentY", "nextAgentTime",
        "avgAgentX", "avgAgentY", "agentSide", "agentRole", "agentObserved", "goalDiff", "eventX", "eventY",
        "eventType", "observeEventX", "observeEventY", "prevAvgX", "prevAvgY", "nextAvgX", "nextAvgY",
        "possessRatio", "elapsedTime"]
CAT_IDX = [i for i, c in enumerate(sorted(XFNS)) if c in config.categorical_features]
CFG = json.load(open("imputer_params.json"))["transformer"]["ModelConfig"]
CFG["dropout"] = 0.0  # CPU에서 dropout=0.1이 약 11배 느린 것이 이 프로젝트에서 측정됨


def make_x(B, W, N):
    X = torch.randn(B, W, N, len(XFNS))
    for i, c in enumerate(sorted(XFNS)):
        if i in CAT_IDX:
            X[..., i] = torch.randint(0, config.categorical_features[c] + 1, (B, W, N)).float()
    return X


def build(dim_output, **over):
    cfg = dict(CFG, **over)
    return M.SetTransformer(dim_input=len(XFNS), num_outputs=len(XFNS), dim_output=dim_output,
                            categorical_indices=CAT_IDX, **cfg).eval()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        check.failed += 1
check.failed = 0


B, W = 2, 5
t0 = time.time()


def make_inputs(B, W, N, p_visible=0.75):
    """X, obs_xy (normalised units), visible. 가려진 선수의 obs_xy는 NaN으로 채워 모델이 읽지 못하는지 함께 확인함."""
    X = make_x(B, W, N)
    obs = torch.rand(B, W, N, 2)
    vis = torch.rand(B, W, N) < p_visible
    obs = torch.where(vis.unsqueeze(-1), obs, torch.full_like(obs, float("nan")))
    return X, obs, vis


print("== 1. forward/backward: varying player counts N, all-visible / all-hidden / mixed, W=5 and W=1")
for dim_out in (2, 4, 6):
    m = build(dim_out)
    print(f"dim_output={dim_out}: total params {n_params(m):,}")
    for Wn in (5, 1):
        for N in (5, 14, 22, 40):
            for kind, p_vis in (("all-visible", 1.0), ("all-hidden", 0.0), ("mixed", 0.75)):
                X, obs, vis = make_inputs(B, Wn, N, p_vis)
                out = m(X, obs, vis)
                ok = out.shape == (B, Wn, N, dim_out) and torch.isfinite(out).all().item()
                if Wn == 5 and N in (5, 40) or (Wn == 1 and N == 22):
                    check(f"W={Wn} N={N:2d} {kind:11s}: out {tuple(out.shape)}, finite", ok)
                elif not ok:
                    check(f"W={Wn} N={N:2d} {kind}", ok)
    m.train()
    m.zero_grad()
    X, obs, vis = make_inputs(B, W, 22)
    out = m(X, obs, vis, torch.ones(B, W, 22, dtype=torch.bool))
    out.pow(2).mean().backward()
    no_grad = sorted({n.split(".")[0] for n, p in m.named_parameters() if p.grad is None})
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"  params without grad: {no_grad}")
    check("backward: all grads finite", not bad)
    check("every parameter gets a gradient (dim_output=2 leaves the unused fc_v)", no_grad == (["fc_v"] if dim_out == 2 else []))
    m.eval()

print("== 2. removed eq (4)(5)(6) path")
m = build(6)
gone = [n for n in ("snapshot_encoder1", "snapshot_encoder2", "snapshot_encoder3", "freeze_fc") if hasattr(m, n)]
check(f"no snapshot_encoder1/2/3, freeze_fc in the model (found: {gone})", not gone)

print("== 3. a hidden token cannot read its coordinate; a visible token can")
m = build(6)
X, obs, vis = make_inputs(B, W, 22)  # 가려진 곳은 NaN
vis[:, :, :4] = True
vis[:, :, 4:8] = False
obs = torch.where(vis.unsqueeze(-1), torch.nan_to_num(obs), torch.full_like(obs, float("nan")))  # 보이는 곳은 유한값, 가려진 곳은 NaN
with torch.no_grad():
    ref = m(X, obs, vis)
    obs2 = torch.where(vis.unsqueeze(-1), obs, torch.full_like(obs, 123.0))  # 가려진 선수의 좌표를 NaN -> 123으로 바꿈
    alt = m(X, obs2, vis)
    obs3 = obs.clone()
    obs3[:, :, 0] += 0.3  # visible 선수 0의 좌표를 바꿈
    changed = m(X, obs3, vis)
check("NaN in hidden obs_xy: output finite", torch.isfinite(ref).all().item())
check(f"hidden obs_xy value (NaN vs 123) does not change the output (max diff {(ref - alt).abs().max().item():.1e})", torch.equal(ref, alt))
check("visible obs_xy does change the output (control)", (ref - changed).abs().max().item() > 1e-3)
with torch.no_grad():
    allvis = m(X, torch.nan_to_num(obs), torch.ones_like(vis))
    allhid = m(X, torch.nan_to_num(obs), torch.zeros_like(vis))
check("same X, all-visible vs all-hidden gives different predictions (visible flag is used)", (allvis - allhid).abs().max().item() > 1e-3)

print("== 4. residual gating: x_hat = x_obs + delta only where visible")
m = build(6)
last = m.fc_xy[-1]
nn.init.zeros_(last.weight)
nn.init.zeros_(last.bias)  # delta == 0
X, obs, vis = make_inputs(B, W, 22)
obs = torch.nan_to_num(obs)
obs[:, :, 0] = torch.tensor([0.93, 0.88])  # 원점에서 먼 좌표
vis[:, :, 0] = True
vis[:, :, 1] = False
with torch.no_grad():
    out = m(X, obs, vis)
xy = out[..., 2:4]
check(f"delta=0: visible player 0 is predicted at x_obs ({xy[0, 0, 0].tolist()}), not near the origin",
      torch.allclose(xy[:, :, 0], obs[:, :, 0]))
check("delta=0: every visible player equals x_obs", torch.equal(xy[vis], obs[vis]))
check("delta=0: every hidden player gets no x_obs term (== 0)", bool((xy[~vis] == 0).all()))
m = build(6)  # 학습 전 임의 초기화에서도: visible 선수 0의 예측은 x_obs=(0.93, 0.88) 쪽으로 편향되어 있어야 함 -> delta 크기만큼 차이
with torch.no_grad():
    delta_scale = m(X, torch.full_like(obs, 0.0), torch.zeros_like(vis))[..., 2:4].abs().mean().item()
    out = m(X, obs, vis)[..., 2:4]
print(f"  random init: mean |xy| of a hidden-only prediction {delta_scale:.3f}")
d_obs, d_origin = (out[:, :, 0] - obs[:, :, 0]).norm(dim=-1).mean().item(), out[:, :, 0].norm(dim=-1).mean().item()
print(f"  random init, visible player 0 with x_obs=(0.93, 0.88): mean distance to x_obs {d_obs:.3f}, to the origin {d_origin:.3f}")
check("random init: visible player 0 is predicted closer to x_obs than to the origin", d_obs < d_origin)

print("== 5. valid mask: 7 real + 15 garbage-padded == 7 real alone")
m = build(6)
Xr, obsr, visr = make_inputs(B, W, 7)
Xg, obsg, visg = make_inputs(B, W, 15)
Xg = Xg * 1.0
obsg = torch.nan_to_num(obsg) * 10
Xp, obsp, visp = torch.cat([Xr, Xg], 2), torch.cat([torch.nan_to_num(obsr), obsg], 2), torch.cat([visr, torch.ones_like(visg)], 2)
valid = torch.cat([torch.ones(B, W, 7), torch.zeros(B, W, 15)], dim=2).bool()
with torch.no_grad():
    ref = m(Xr, obsr, visr)
    masked = m(Xp, obsp, visp, valid)[:, :, :7]
    unmasked = m(Xp, obsp, visp)[:, :, :7]
d_masked, d_unmasked = (ref - masked).abs().max().item(), (ref - unmasked).abs().max().item()
print(f"  max|ref - masked| = {d_masked:.2e}   max|ref - unmasked| = {d_unmasked:.2e}")
check("masked padding does not change the output (<1e-4)", d_masked < 1e-4)
check("unmasked padding does change the output (control)", d_unmasked > 1e-3)
with torch.no_grad():
    check("all-invalid snapshot stays finite", torch.isfinite(m(Xr, obsr, visr, torch.zeros(B, W, 7, dtype=torch.bool))).all().item())

print("== 5b. valid mask on the time axis: garbage in absent (player, timestep) tokens does not change valid outputs")
m = build(6)
X, obs, vis = make_inputs(B, W, 12)
obs = torch.nan_to_num(obs)
valid = torch.ones(B, W, 12, dtype=torch.bool)
valid[:, 0, :5] = False  # 선수 0~4가 첫 timestep에 없음
valid[:, 3:, 5] = False  # 선수 5가 마지막 두 timestep에 없음
X2, obs2 = X.clone(), obs.clone()
X2[~valid] = torch.randn(int((~valid).sum()), X.shape[-1]) * 5
for i, c in enumerate(sorted(XFNS)):
    if i in CAT_IDX:
        X2[..., i] = torch.where(valid, X2[..., i], torch.zeros_like(X2[..., i]))
obs2[~valid] = 9.0
with torch.no_grad():
    o1, o2 = m(X, obs, vis, valid), m(X2, obs2, vis, valid)
d = (o1 - o2)[valid].abs().max().item()
check(f"garbage in invalid tokens leaves valid outputs unchanged (max diff {d:.1e} < 1e-4)", d < 1e-4)
with torch.no_grad():
    d_ctrl = (m(X, obs, vis)[valid] - m(X2, obs2, vis)[valid]).abs().max().item()
check(f"control: without the valid mask the same garbage does change valid outputs (max diff {d_ctrl:.1e})", d_ctrl > 1e-3)

print("== 6. loss: standardised units, Huber, visible/hidden split, heading mask")
T = M.TARGET_SCALE
cols = ("vx", "vy", "x", "y", "ax", "ay")
tscale = torch.tensor([T[c] for c in cols])


def loss_inputs(Bn=4, Nn=20, p_vis=0.75, D=6):
    t = torch.randn(Bn, W, Nn, 6) * tscale
    v = torch.rand(Bn, W, Nn) < p_vis
    return t[..., :D] if D == 6 else t[..., :D], v


def bl(p, t, v, valid=None):
    return M.balanced_loss(p, t, None, v, valid)


# 6a. 위치 오차 1 표준편차 하나만 있을 때 정확한 값: 0.7 * mean(huber(1)=0.5, huber(0)=0) = 0.175 (속도는 완벽히 일치)
t, v = loss_inputs(D=4)
t[..., :2] = torch.tensor([3.0, 1.0])  # 모두 움직임
p = t.clone()
p[..., 2] += T["x"]
_, loss, eu, _ = bl(p, t, torch.ones_like(v))
check(f"6a. 1-std x error -> loss {loss.item():.4f} == W_POS*0.25 = {M.LOSS_W_POS * 0.25:.4f}", abs(loss.item() - M.LOSS_W_POS * 0.25) < 1e-5)
check(f"6a. euclid is reported in metres: {eu.item():.2f} == {T['x'] * 105:.2f}", abs(eu.item() - T["x"] * config.field_length) < 1e-3)

# 6b. Huber: 큰 오차는 선형으로 증가 (MSE였다면 제곱)
p2 = t.clone()
p2[..., 2] += 10 * T["x"]
_, loss10, _, _ = bl(p2, t, torch.ones_like(v))
check(f"6b. 10-std error -> {loss10.item():.3f} == W_POS*0.5*(10-0.5) = {M.LOSS_W_POS * 0.5 * 9.5:.3f} (linear tail, MSE would give {M.LOSS_W_POS * 0.5 * 50:.2f})",
      abs(loss10.item() - M.LOSS_W_POS * 0.5 * 9.5) < 1e-4)

# 6c. visible/hidden 분리: visible은 완벽, hidden만 오차 -> 단일 pooled 평균(약 hidden 비율 25%)의 약 2배
t, v = loss_inputs(Bn=8, Nn=20, p_vis=0.75)
p = t.clone()
p[~v] += torch.randn(int((~v).sum()), 6) * tscale
_, loss, _, _ = bl(p, t, v)
pooled = M._masked_loss(p.reshape(-1, 6), t.reshape(-1, 6), M._COMPONENTS[6])
l_h = M._masked_loss(p[:, W // 2][~v[:, W // 2]], t[:, W // 2][~v[:, W // 2]], M._COMPONENTS[6])
l_v = M._masked_loss(p[:, W // 2][v[:, W // 2]], t[:, W // 2][v[:, W // 2]], M._COMPONENTS[6])
expected = (M.LOSS_W_VISIBLE * l_v + M.LOSS_W_HIDDEN * l_h) / (M.LOSS_W_VISIBLE + M.LOSS_W_HIDDEN)
frac_h = (~v[:, W // 2]).float().mean().item()
print(f"  hidden fraction {frac_h:.2f}; loss {loss.item():.4f}; l_visible {l_v.item():.4f}; l_hidden {l_h.item():.4f}; pooled {pooled.item():.4f}")
check("6c. loss == (W_VISIBLE*L_visible + W_HIDDEN*L_hidden)/(W_VISIBLE+W_HIDDEN)", abs(loss.item() - expected.item()) < 1e-5)
check("6c. visible-perfect case: L_visible is 0 (cos/Huber) and the split loss exceeds the pooled mean", l_v.item() < 1e-6 and loss.item() > 1.5 * pooled.item())

# 6d. 한쪽 부분집합이 빈 배치: 남은 쪽만 사용, finite
for name, vv in (("all-visible", torch.ones_like(v)), ("all-hidden", torch.zeros_like(v))):
    p = t + torch.randn_like(t) * tscale * 0.3
    _, loss, eu, _ = bl(p, t, vv)
    ref = M._masked_loss(p[:, W // 2].reshape(-1, 6), t[:, W // 2].reshape(-1, 6), M._COMPONENTS[6])
    check(f"6d. {name}: loss {loss.item():.4f} == single-subset loss {ref.item():.4f}", abs(loss.item() - ref.item()) < 1e-5 and torch.isfinite(eu).item())

# 6e. heading mask: 정답 속력 < HEADING_MIN_SPEED 이면 cosine 항이 loss에서 빠짐
def heading_effect(speed):
    t = torch.zeros(1, W, 2, 4)
    t[..., 0] = speed
    p = t.clone()
    p[..., 0] = -speed  # 정반대 방향
    ones = torch.ones(1, W, 2, dtype=torch.bool)
    l_with = bl(p, t, ones)[1].item()
    old = M.LOSS_W_COS
    M.LOSS_W_COS = 0.0
    l_without = bl(p, t, ones)[1].item()
    M.LOSS_W_COS = old
    return l_with - l_without
d_slow, d_fast = heading_effect(0.5 * M.HEADING_MIN_SPEED), heading_effect(2.0 * M.HEADING_MIN_SPEED)
print(f"  HEADING_MIN_SPEED={M.HEADING_MIN_SPEED} m/s: cosine contribution at speed {0.5 * M.HEADING_MIN_SPEED:.2f} = {d_slow:.4f}, at {2.0 * M.HEADING_MIN_SPEED:.2f} = {d_fast:.4f}")
check("6e. slow token (opposite heading): cosine term is masked (0)", abs(d_slow) < 1e-9)
check(f"6e. fast token (opposite heading): cosine term = W_COS*2 = {M.LOSS_W_COS * 2:.2f}", abs(d_fast - M.LOSS_W_COS * 2) < 1e-6)

# 6f. NaN 라벨(뛰지 않는 선수)과 valid=False는 loss에서 제외, 결과는 finite
t, v = loss_inputs(Bn=4, Nn=20)
p = t + torch.randn_like(t) * tscale * 0.3
valid = torch.ones_like(v)
valid[:, :, 15:] = False
t_nan = t.clone()
t_nan[:, :, 10:12] = float("nan")
t_nan_pad = t_nan.clone()
t_nan_pad[:, :, 15:] = 1e6  # padding 슬롯의 쓰레기 라벨
keep = list(range(10)) + [12, 13, 14]  # NaN(10, 11)과 padding(15 이상)을 뺀 선수
_, l_ref, _, _ = bl(p[:, :, keep], t_nan[:, :, keep], v[:, :, keep])
_, l_msk, _, _ = bl(p, t_nan_pad, v, valid)
check(f"6f. NaN labels and valid=False are excluded ({l_msk.item():.4f} == {l_ref.item():.4f})", abs(l_msk.item() - l_ref.item()) < 1e-5)

# 6g. 기준선 오차에서 위치 항과 속도 항의 크기 비교 (성분당 오차: 위치 2 m, 속도 1.9 m/s, 방향 항 제외)
def term_ratio(scale):
    old_scale, old = dict(M.TARGET_SCALE), (M.LOSS_W_VEL, M.LOSS_W_POS, M.LOSS_W_COS, M.LOSS_W_ACC)
    M.TARGET_SCALE.update(scale)
    M.LOSS_W_COS = 0.0
    n = 1000
    tt = torch.zeros(n, 4)
    pp = tt.clone()
    sign = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    pp[:, 0] += 1.9 * sign
    pp[:, 1] += 1.9 * sign
    pp[:, 2] += 2.0 / config.field_length * sign
    pp[:, 3] += 2.0 / config.field_width * sign
    M.LOSS_W_VEL = 0.0
    pos_term = M._masked_loss(pp, tt, M._COMPONENTS[4]).item()
    M.LOSS_W_VEL, M.LOSS_W_POS = old[0], 0.0
    vel_term = M._masked_loss(pp, tt, M._COMPONENTS[4]).item()
    M.TARGET_SCALE.clear(); M.TARGET_SCALE.update(old_scale)
    M.LOSS_W_VEL, M.LOSS_W_POS, M.LOSS_W_COS, M.LOSS_W_ACC = old
    return pos_term, vel_term
before = term_ratio({"vx": 2.3, "vy": 1.6, "x": 0.23, "y": 0.23})  # 한계 표준편차 scale (이전 값)
after = term_ratio({})
print(f"  baseline errors (2 m, 1.9 m/s), weighted terms  before: pos {before[0]:.4f} vel {before[1]:.4f} ratio pos:vel = 1:{before[1] / before[0]:.1f}"
      f"   after: pos {after[0]:.4f} vel {after[1]:.4f} ratio pos:vel = {after[0] / after[1]:.2f}:1  (weights alone: {M.LOSS_W_POS / M.LOSS_W_VEL:.2f}:1)")
ratio_after, ratio_w = after[0] / after[1], M.LOSS_W_POS / M.LOSS_W_VEL
check(f"6g. pos:vel at baseline error {ratio_after:.2f}:1 is within 15% of the weight ratio {ratio_w:.2f}:1", abs(ratio_after / ratio_w - 1) < 0.15)

# 6h. 가속도 가중치를 0으로 두면 D=6 loss가 가속도 열을 무시하고 속도/위치/방향 항은 그대로
t, v = loss_inputs(Bn=4, Nn=20, p_vis=0.75)
p = t + torch.randn_like(t) * tscale * 0.5
old_acc = M.LOSS_W_ACC
M.LOSS_W_ACC = 0.0
l6 = bl(p, t, v)[1].item()
p_alt = p.clone()
p_alt[..., 4:] += 5.0  # 가속도 예측만 크게 바꿈
l6_alt = bl(p_alt, t, v)[1].item()
M.LOSS_W_ACC = old_acc
l_acc_on = bl(p_alt, t, v)[1].item()
check(f"6h. LOSS_W_ACC=0: acceleration errors do not change the loss ({l6:.6f} == {l6_alt:.6f}); with LOSS_W_ACC={old_acc} they do ({l_acc_on:.4f})",
      abs(l6 - l6_alt) < 1e-6 and l_acc_on > l6 + 1e-3)

print("== 7. SetTransformerModel.step (fake dataset), D=2/4/6")
class FakeDataset:
    def __getitem__(self, i):
        return {"features": make_x(1, W, 22)[0], "labels": torch.randn(W, 22, DIM), "categorical_indices": CAT_IDX,
                "actor_global_index_lst": 0}
for DIM in (2, 4, 6):
    lm = M.SetTransformerModel(FakeDataset(), dict(CFG), {"optimizer_params": {"lr": 1e-4}})
    lm.eval()
    labels = torch.randn(B, W, 22, DIM)
    labels[:, :, 20:] = float("nan")  # 뛰지 않는 선수
    X, obs, vis = make_inputs(B, W, 22)
    batch = {"features": X, "labels": labels, "actor_global_index_lst": torch.zeros(B, dtype=torch.long),
             "obs_xy": obs, "visible": vis}
    mae, loss, eu, cos = lm.step(batch)
    loss.backward()
    print(f"  dim={DIM}: loss={loss.item():.4f} euclid={eu.item():.2f} cos={cos.item():.4f}")
    check(f"dim={DIM}: loss finite, backward ok", torch.isfinite(loss).item())

print("== 8. layers are independent across depth")
layer = M.CustomTransformerEncoderLayer(32, 4, 64, 0.0)
enc3 = M.CustomTransformerEncoder(layer, 3)
n_layer = n_params(layer)
check("CustomTransformerEncoder: layers[0] is not layers[1] is not layers[2]",
      enc3.layers[0] is not enc3.layers[1] and enc3.layers[1] is not enc3.layers[2] and enc3.layers[0] is not enc3.layers[2])
check(f"CustomTransformerEncoder: params {n_params(enc3):,} == 3 x layer {n_layer:,} + norm {n_params(enc3.norm):,}",
      n_params(enc3) == 3 * n_layer + n_params(enc3.norm))
with torch.no_grad():
    enc3.layers[0].linear1.weight.add_(1.0)
check("CustomTransformerEncoder: editing layer 0 leaves layer 1 untouched",
      not torch.equal(enc3.layers[0].linear1.weight, enc3.layers[1].linear1.weight))
mods = {}
for nl in (1, 2, 3):
    mods[nl] = build(6, num_layers=nl)
per_layer = n_params(mods[2]) - n_params(mods[1])
one_set = sum(n_params(l) for l in (mods[1].player_encoder1.layers[0], mods[1].time_encoder1.layers[0]))
print(f"  SetTransformer params num_layers=1/2/3: {n_params(mods[1]):,} / {n_params(mods[2]):,} / {n_params(mods[3]):,}")
check("SetTransformer: player_encoder1 / time_encoder1 layers are distinct objects",
      mods[2].player_encoder1.layers[0] is not mods[2].player_encoder1.layers[1] and mods[2].time_encoder1.layers[0] is not mods[2].time_encoder1.layers[1])
check(f"SetTransformer: each extra layer adds {one_set:,} params (player + time layer), linearly",
      per_layer == one_set and n_params(mods[3]) - n_params(mods[2]) == one_set)

print("== 9. dead classes are not instantiated by the live path")
live = sorted({type(x).__name__ for x in build(4).modules()})
print("  module types in SetTransformer:", live)
dead = ["MAB", "SAB", "PMA", "ISAB", "MLP", "AgentAwareAttention", "AgentAwareBlock", "CustomTransformerEncoder"]
check(f"none of {dead} is instantiated", not any(d in live for d in dead))

print(f"done in {time.time() - t0:.1f}s, failures: {check.failed}")
sys.exit(1 if check.failed else 0)
