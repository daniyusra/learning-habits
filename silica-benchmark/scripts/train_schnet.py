"""Train one SchNet learning-curve point with SchNetPack 2.2, then score it on the frozen test set.
Runs in silica-benchmark/.venv-schnet (Python 3.12).

    silica-benchmark/.venv-schnet/bin/python silica-benchmark/scripts/train_schnet.py SEED N [--epochs E] [--tag _pilot]

SchNet: 128 features, 6 interactions, 50 Gaussians, cosine cutoff 6 Å (same cutoff as MACE). E0s = SCAN isolated atoms.
Same validation set, batch size and epoch schedule as train_mace.py; SchNet validates every epoch (it is cheap),
with patience scaled by MACE's evaluation interval so early stopping covers the same number of epochs. Loss weights energy 0.01 / forces 1 / stress 10
(SchNetPack's energy loss is on total energy, so the weights are not identical to MACE's per-atom ones).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
from ase.io import read

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402
from train_mace import SCAN_E0, SCHEDULE  # noqa: E402  (pure constants, no MACE import at module level)

ROOT = Path("silica-benchmark").resolve()
CUTOFF = 6.0


class MAE(torchmetrics.MeanAbsoluteError):
    """torchmetrics calls .view(-1), which fails on the non-contiguous stress that Strain produces."""

    def update(self, preds, target):
        super().update(preds.contiguous(), target.contiguous())


def atomrefs():
    ref = torch.zeros(100)
    for z, e in SCAN_E0.items():
        ref[z] = e
    return ref


def build_db(path, atoms_list):
    if path.exists():
        path.unlink()
    db = spk.data.ASEAtomsData.create(str(path), distance_unit="Ang",
                                      property_unit_dict={"energy": "eV", "forces": "eV/Ang", "stress": "eV/Ang/Ang/Ang"},
                                      atomrefs={"energy": atomrefs().tolist()})
    props = [{"energy": np.array([a.info["REF_energy"]]), "forces": a.arrays["REF_forces"],
              "stress": np.asarray(a.info["REF_stress"], dtype=float).reshape(1, 3, 3)} for a in atoms_list]
    plain = []
    for a in atoms_list:
        b = a.copy()
        b.info, b.calc = {}, None
        for k in list(b.arrays):
            if k not in ("numbers", "positions"):
                del b.arrays[k]
        plain.append(b)
    db.add_systems(property_list=props, atoms_list=plain)


def build_model():
    schnet = spk.representation.SchNet(n_atom_basis=128, n_interactions=6,
                                       radial_basis=spk.nn.GaussianRBF(n_rbf=50, cutoff=CUTOFF),
                                       cutoff_fn=spk.nn.CosineCutoff(CUTOFF))
    return spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[spk.atomistic.Strain(), spk.atomistic.PairwiseDistances()],
        output_modules=[spk.atomistic.Atomwise(n_in=128, output_key="energy"),
                        spk.atomistic.Forces(energy_key="energy", force_key="forces", stress_key="stress", calc_stress=True)],
        postprocessors=[trn.CastTo64(), trn.AddOffsets("energy", add_mean=True, add_atomrefs=True)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    name = f"s{args.seed}_n{args.n}{args.tag}"
    work = ROOT / "runs" / "schnet" / name
    work.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    train = read(ROOT / f"data/lc/s{args.seed}_n{args.n}.xyz", index=":")
    val = read(ROOT / "data/lc/val.xyz", index=":")
    max_epochs, eval_every = SCHEDULE[args.n]
    if args.epochs:
        max_epochs = args.epochs

    db = work / "data.db"
    build_db(db, train + val)
    split = work / "split.npz"
    n = len(train)
    np.savez(split, train_idx=np.arange(n), val_idx=np.arange(n, n + len(val)), test_idx=np.arange(n, n + 1))
    dm = spk.data.AtomsDataModule(
        str(db), batch_size=2, val_batch_size=2, split_file=str(split), num_workers=0, pin_memory=True,
        load_properties=["energy", "forces", "stress"],
        transforms=[trn.MatScipyNeighborList(cutoff=CUTOFF),
                    trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=True),
                    trn.CastTo32()],
    )
    dm.prepare_data()
    dm.setup()

    model = build_model()
    outputs = [spk.task.ModelOutput("energy", loss_fn=torch.nn.MSELoss(), loss_weight=0.01, metrics={"MAE": MAE()}),
               spk.task.ModelOutput("forces", loss_fn=torch.nn.MSELoss(), loss_weight=1.0, metrics={"MAE": MAE()}),
               spk.task.ModelOutput("stress", loss_fn=torch.nn.MSELoss(), loss_weight=10.0, metrics={"MAE": MAE()})]
    task = spk.task.AtomisticTask(model=model, outputs=outputs, optimizer_cls=torch.optim.AdamW,
                                  optimizer_args={"lr": 5e-4},
                                  scheduler_cls=spk.train.ReduceLROnPlateau,
                                  scheduler_args={"factor": 0.8, "patience": 5 * eval_every, "min_lr": 1e-6},
                                  scheduler_monitor="val_loss")
    best = work / "best_model"
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs, check_val_every_n_epoch=1,
                         default_root_dir=str(work), enable_progress_bar=False,
                         logger=pl.loggers.CSVLogger(str(work), name="log"),
                         callbacks=[spk.train.ModelCheckpoint(model_path=str(best), save_top_k=1, monitor="val_loss"),
                                    pl.callbacks.EarlyStopping(monitor="val_loss", patience=10 * eval_every)])
    t0 = time.time()
    trainer.fit(task, datamodule=dm)
    train_s = time.time() - t0

    calc = spk.interfaces.SpkCalculator(model=str(best), neighbor_list=trn.MatScipyNeighborList(cutoff=CUTOFF),
                                        energy_key="energy", force_key="forces", stress_key="stress",
                                        energy_unit="eV", position_unit="Ang", device="cuda", dtype=torch.float32)
    test = read(ROOT / "data/splits/test.xyz", index=":")
    E, F, S = [], [], []
    for a in test:
        b = a.copy()
        b.calc = calc
        E.append(b.get_potential_energy())
        F.append(b.get_forces())
        S.append(b.get_stress(voigt=False))
    out = ROOT / "results/schnet" / f"{name}.json" if not args.tag else work / "result.json"
    metrics.save(out, "schnet", args.seed, args.n, metrics.score(test, E, F, S),
                 {"train_seconds": round(train_s), "n_train_atoms": sum(len(a) for a in train),
                  "max_epochs": max_epochs, "epochs_run": trainer.current_epoch,
                  "n_params": sum(p.numel() for p in model.parameters())})


if __name__ == "__main__":
    main()
