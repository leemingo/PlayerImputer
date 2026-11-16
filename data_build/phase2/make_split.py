"""Deterministic match-level split. Writes ../out/splits.csv."""
import hashlib, os, pandas as pd
ROOT = "/hpcwork/p0029161/scoutgpt/_cdfbatch/out/2026-2027"
EXCLUDE_NO_PLAYER_ID = {4077658, 4077659, 4077660, 4077677, 4077679}
WEAK_ALIGN = {4077636, 4077638, 4077641, 4077644, 4077656, 4077667, 4077670, 4077678}
# (train, valid, test) match counts per stratum
QUOTA = {True: (6, 1, 1), False: (34, 7, 7)}   # key = align_weak

recs = []
for d in sorted(os.listdir(ROOT)):
    sb, dfl = d.split("_", 1)
    sb = int(sb)
    if sb in EXCLUDE_NO_PLAYER_ID:
        continue
    recs.append(dict(match_id=sb, dfl_match_id=dfl, align_weak=sb in WEAK_ALIGN,
                     hash=hashlib.sha256(f"phase2-v1:{sb}".encode()).hexdigest()))
df = pd.DataFrame(recs)
assert len(df) == 56, len(df)
df["split"] = ""
for weak, (a, b, c) in QUOTA.items():
    idx = df[df.align_weak == weak].sort_values("hash").index
    assert len(idx) == a + b + c, (weak, len(idx))
    df.loc[idx[:a], "split"] = "train"
    df.loc[idx[a:a + b], "split"] = "valid"
    df.loc[idx[a + b:], "split"] = "test"
df = df.sort_values("match_id").reset_index(drop=True)
df.drop(columns="hash").to_csv("/hpcwork/p0029161/scoutgpt/_phase2_dataset/out/splits.csv", index=False)
print(df.groupby(["split", "align_weak"]).size())
