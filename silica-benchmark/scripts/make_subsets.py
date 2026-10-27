"""Write the learning-curve subsets from split_v1.json.

Run from the repo root with any env that has ASE:
    .venv/bin/python silica-benchmark/scripts/make_subsets.py

Outputs (git-ignored, regenerable):
    silica-benchmark/data/lc/val.xyz          fixed 50-structure validation set, shared by every run
    silica-benchmark/data/lc/s{seed}_n{N}.xyz training subset: first N of that seed's ordering, val removed
Tracked:
    silica-benchmark/lc_plan_v1.json          the validation indices and the step-1 grid
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path("silica-benchmark")
N_VAL = 50
SIZES = [10, 30, 100, 300]
SEEDS = [0, 1, 2]

split = json.loads((ROOT / "split_v1.json").read_text())
pool = {a.info["orig_index"]: a for a in read(ROOT / "data/splits/pool.xyz", index=":")}

val_idx = sorted(int(i) for i in np.random.default_rng(7).choice(split["pool"], size=N_VAL, replace=False))
out = ROOT / "data/lc"
out.mkdir(parents=True, exist_ok=True)
write(out / "val.xyz", [pool[i] for i in val_idx], format="extxyz")

for seed in SEEDS:
    order = [i for i in split["orders"][str(seed)] if i not in set(val_idx)]
    for n in SIZES:
        write(out / f"s{seed}_n{n}.xyz", [pool[i] for i in order[:n]], format="extxyz")

plan = {"split": "split_v1.json", "val": val_idx, "sizes": SIZES, "seeds": SEEDS,
        "note": "training subset = first N of split_v1 orders[seed] after removing val; nested across N"}
(ROOT / "lc_plan_v1.json").write_text(json.dumps(plan))
fams = {}
for i in val_idx:
    fams[pool[i].info["family"]] = fams.get(pool[i].info["family"], 0) + 1
print(f"val: {N_VAL} structures, {sum(len(pool[i]) for i in val_idx)} atoms, families {fams}")
print(f"wrote {len(SEEDS) * len(SIZES)} training subsets to {out}")
