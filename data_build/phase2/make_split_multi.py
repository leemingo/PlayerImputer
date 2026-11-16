"""Multi-competition match selection and split. Writes splits.csv and splits_summary.json to OUT.

Supersedes make_split.py for the multi-competition build; make_split.py stays as the record of the
frozen 56-match 1. Bundesliga build. Competition 9 keeps that build's split unchanged.
"""
import hashlib, json, os, pandas as pd
ROOT = "/hpcwork/p0029161/scoutgpt/_cdfbatch_multi/out/2026-2027"   # <sb>_<dfl>/{alignment.json,sb360_objects.parquet}
TARGETS = "/hpcwork/p0029161/scoutgpt/_dfl_download_20261116/results/target_matches_2627.csv"   # comp,sb,dfl,date
OLD_SPLITS = "/hpcwork/p0029161/scoutgpt/_phase2_dataset/out/splits.csv"   # frozen 56-match build
OUT = "/hpcwork/p0029161/scoutgpt/_phase2_multileague"
RECOVERED = {4077658, 4077659, 4077660, 4077677, 4077679}   # competition 9, absent from TARGETS
EXPECTED_COMP9 = 79   # 56 frozen + 20 new + 5 recovered = 81, minus 4077237 and 4077657 (5.6% player_id fill, rule 1)
# Selection thresholds (measured on the 342 matches of competitions 10/135/165/179).
MIN_PLAYER_ID_FILL = 0.5   # drops 68 of 342; the dropped ones sit at 6-10% fill (actor only)
BAD_FLAGS = {"BOUNDARY", "NOT_V_SHAPED", "AMBIGUOUS_MINIMUM"}   # offset not identified; drops 21
MAX_MEDIAN_M = 8.0   # drops 2 more; without rules 2 and 3 the worst match is 19.78 m, with them 7.15 m
WEAK_MEDIAN_M = 4.12   # align_weak threshold of the 56-match build; 61 of 251 survivors; still trained on
# Train/valid/test shares per competition (stratified by align_weak, hash ordered).
SHARES = {10: (.70, .15, .15), 135: (.70, .15, .15),
          179: (0, .15, .85), 165: (0, .15, .85)}   # held out of every training regime: train share goes to test
rnd = lambda x: int(x + 0.5)

old = pd.read_csv(OLD_SPLITS)
assert len(old) == 56 and old.split.value_counts().loc[["valid", "test"]].eq(8).all()
tg = pd.read_csv(TARGETS)
comp = dict(zip(tg.sb, tg.comp)); comp.update({m: 9 for m in RECOVERED})
done = set(old.match_id)
missing = sorted(set(comp) - {int(d.split("_")[0]) for d in os.listdir(ROOT)})

recs, drops = [], {}
for d in sorted(os.listdir(ROOT)):
    sb, dfl = d.split("_", 1); sb = int(sb)
    al = json.load(open(f"{ROOT}/{d}/alignment.json"))["alignment"]
    fill = pd.read_parquet(f"{ROOT}/{d}/sb360_objects.parquet", columns=["player_id"]).player_id.notna().mean()
    flags = {f for p in al["periods"].values() for f in p["flags"]}
    worst = max(p["median_m"] for p in al["periods"].values())
    reason = ("player_id_fill" if fill < MIN_PLAYER_ID_FILL else
              "alignment_flag" if flags & BAD_FLAGS else
              "median_m" if worst > MAX_MEDIAN_M else None)
    if reason:
        drops.setdefault(comp[sb], {}).setdefault(reason, []).append(sb)
        continue
    recs.append(dict(match_id=sb, dfl_match_id=dfl, competition=comp[sb], align_weak=worst > WEAK_MEDIAN_M,
                     hash=hashlib.sha256(f"phase3-{comp[sb]}-v1:{sb}".encode()).hexdigest(), split=""))
df = pd.DataFrame(recs)

# Competition 9: frozen 56 carried over, every additional match is train.
df.loc[df.competition == 9, "split"] = "train"
frozen = old.assign(competition=9)[["match_id", "dfl_match_id", "competition", "align_weak", "split"]]
# Other competitions: per align_weak stratum, hash ordered.
for c, (ptr, pva, _) in SHARES.items():
    for weak in (True, False):
        idx = df[(df.competition == c) & (df.align_weak == weak)].sort_values("hash").index
        a = rnd(ptr * len(idx)); b = rnd(pva * len(idx))
        df.loc[idx[:a], "split"] = "train"; df.loc[idx[a:a + b], "split"] = "valid"; df.loc[idx[a + b:], "split"] = "test"
out = pd.concat([frozen, df.drop(columns="hash")]).sort_values("match_id").reset_index(drop=True)

assert out.match_id.is_unique and (out.split != "").all()
kept = out[out.match_id.isin(done)].set_index("match_id").split
assert kept.equals(old.set_index("match_id").split.reindex(kept.index)) and len(kept) == 56   # frozen test set must not move
c9 = out[out.competition == 9]
assert (c9.split == "valid").sum() == 8 and (c9.split == "test").sum() == 8
assert len(c9) == EXPECTED_COMP9, (len(c9), "unconverted:", missing)

os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/splits.csv", index=False)
n = out.groupby(["competition", "split"]).size().unstack(fill_value=0)
summ = {int(c): dict(usable=int(n.loc[c].sum()), **{s: int(n.loc[c, s]) for s in ("train", "valid", "test")},
                     align_weak=int(out[out.competition == c].align_weak.sum()),
                     dropped={r: len(v) for r, v in drops.get(c, {}).items()}) for c in n.index}
summ["unconverted"] = missing   # targets with no output directory
json.dump(summ, open(f"{OUT}/splits_summary.json", "w"), indent=1)
print(n, "\ndropped:", {c: s["dropped"] for c, s in summ.items() if c != "unconverted"}, "\nunconverted:", missing)
