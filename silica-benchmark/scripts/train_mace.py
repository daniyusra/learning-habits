"""Train one MACE learning-curve point, then score it on the frozen test set. Runs in .venv-mace.

    silica-benchmark/.venv-mace/bin/python silica-benchmark/scripts/train_mace.py {ft|scratch} SEED N [--epochs E]

ft      = MACE-MP-0 medium, naive fine-tune (single head, no replay of MP data), E0s = MP-0 E0s + per-element shift
          fitted on the N training structures, lr 1e-3
scratch = the same architecture as MACE-MP-0 medium, random init, E0s = SCAN isolated atoms, lr 1e-2
Both: float32, batch 2 (4 GB GPU limit), loss weights energy 1 / forces 10 / stress 100, EMA, best-on-validation checkpoint.
"""
import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

from ase.io import read

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402
import predict_mace  # noqa: E402

ROOT = Path("silica-benchmark").resolve()
# max epochs and evaluation interval per N: about 1000 / 1500 / 2500 / 4500 gradient steps (batch 2), early stopping on top
SCHEDULE = {10: (200, 5), 30: (100, 2), 100: (50, 1), 300: (30, 1)}
SCAN_E0 = {8: -0.66320916, 14: -4.06855316}
MP0_E0 = {8: -7.2845988273620605, 14: -7.694793224334717}
ARCH = ["--model=MACE", "--r_max=6.0", "--num_radial_basis=10", "--num_cutoff_basis=5", "--max_ell=3",
        "--interaction_first=RealAgnosticResidualInteractionBlock", "--interaction=RealAgnosticResidualInteractionBlock",
        "--num_interactions=2", "--hidden_irreps=128x0e+128x1o", "--MLP_irreps=16x0e", "--correlation=3",
        "--radial_MLP=[64, 64, 64]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("approach", choices=["ft", "scratch"])
    ap.add_argument("seed", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (pilot runs)")
    ap.add_argument("--tag", default="", help="suffix for pilot runs, keeps them out of results/")
    args = ap.parse_args()

    name = f"s{args.seed}_n{args.n}{args.tag}"
    approach = {"ft": "mace_mp0_finetune", "scratch": "mace_scratch"}[args.approach]
    work = ROOT / "runs" / approach / name
    work.mkdir(parents=True, exist_ok=True)
    train_file = ROOT / f"data/lc/s{args.seed}_n{args.n}.xyz"
    train = read(train_file, index=":")
    max_epochs, eval_every = SCHEDULE[args.n]
    if args.epochs:
        max_epochs = args.epochs

    if args.approach == "ft":
        calc = predict_mace.load_calc("medium", dtype="float32")
        E, _, _ = predict_mace.predict(calc, train)
        shift = metrics.fit_element_shift(train, E)
        e0 = {z: MP0_E0[z] + shift[z] for z in MP0_E0}
        del calc
        model_args = ["--foundation_model=medium", "--multiheads_finetuning=False", "--lr=0.001"]
    else:
        e0 = dict(SCAN_E0)
        model_args = ARCH + ["--lr=0.01"]

    cmd = [str(Path(sys.executable).parent / "mace_run_train"), f"--name={approach}", f"--work_dir={work}",
           f"--train_file={train_file}", f"--valid_file={ROOT / 'data/lc/val.xyz'}",
           "--energy_key=REF_energy", "--forces_key=REF_forces", "--stress_key=REF_stress",
           f"--E0s={ {int(k): float(v) for k, v in e0.items()} }",
           "--loss=stress", "--energy_weight=1", "--forces_weight=10", "--stress_weight=100",
           "--batch_size=2", "--valid_batch_size=2", f"--max_num_epochs={max_epochs}", f"--eval_interval={eval_every}",
           "--patience=10", "--ema", "--ema_decay=0.99", "--default_dtype=float32", "--device=cuda",
           f"--seed={args.seed}", "--restart_latest", "--save_cpu", "--error_table=PerAtomMAEstressvirials"] + model_args
    (work / "cmd.txt").write_text(" ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=work, stdout=open(work / "train.log", "a"), stderr=subprocess.STDOUT)
    train_s = time.time() - t0

    model_file = sorted(work.glob(f"{approach}*.model"), key=lambda p: len(p.name))[0]
    test = read(ROOT / "data/splits/test.xyz", index=":")
    E, F, S = predict_mace.predict(predict_mace.load_calc(model_file), test)
    out = ROOT / "results" / approach / f"{name}.json" if not args.tag else work / "result.json"
    metrics.save(out, approach, args.seed, args.n, metrics.score(test, E, F, S),
                 {"train_seconds": round(train_s), "n_train_atoms": sum(len(a) for a in train), "E0s": e0,
                  "max_epochs": max_epochs, "model_file": str(model_file.relative_to(ROOT))})


if __name__ == "__main__":
    main()
