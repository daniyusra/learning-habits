# nequip-md17: training real NequIP (stage 3c)

This folder trains the official **nequip 0.19.1** package on **CCSD(T) ethanol**, the MD17 ethanol trajectory recomputed at the coupled-cluster level. The analysis lives in `../nequip-ethanol.ipynb`, which covers learning curves, test accuracy, symmetry checks, closed-loop work and NVE MD. The from-scratch build that explains each piece is in `../nequip-walkthrough.ipynb`.

## What's here

| Path | What it is | In git? |
|---|---|---|
| `ethanol.yaml` | The training config (Hydra + Lightning). This is the only file you need to reproduce everything. | yes |
| `README.md`, `.gitignore` | This file, and ignore rules for the generated folders | yes |
| `data/` | sGDML CCSD(T) ethanol (`ethanol_ccsd_t-train.npz`, 1000 frames; `-test.npz`, 1000 frames). Downloaded automatically on the first run, ~1.7 MB. | no (root `.gitignore`: `data/`) |
| `outputs/<run>/` | One folder per run: `best.ckpt` (best validation), `last.ckpt`, `train.log`, `lightning_logs/version_0/metrics.csv` (per-epoch metrics), Hydra's config dump | no |
| `outputs/ethanol-lmax1/`, `outputs/ethanol-lmax2/` | The two full runs compared in §6 of the notebook (test force MAE 0.419 vs 0.244 kcal/mol/Å) | no |
| `outputs/smoke/` | A 2-epoch test run, used to check the pipeline | no |
| `*.log` | Console output captured from background runs | no |

Everything ignored can be regenerated from `ethanol.yaml`. If you want to keep a trained model, copy its `best.ckpt` somewhere outside `outputs/`, or run `nequip-package` on it.

## How to run

From the repo root:

```bash
# full run (~70 s/epoch on the GTX 1650 Ti, ~4 h for 200 epochs; stops early after 40 epochs without improvement)
.venv/bin/nequip-train -cp $PWD/nequip-md17 -cn ethanol.yaml hydra.run.dir=nequip-md17/outputs/ethanol-lmax1

# 2-epoch smoke test
.venv/bin/nequip-train -cp $PWD/nequip-md17 -cn ethanol.yaml trainer.max_epochs=2 hydra.run.dir=nequip-md17/outputs/smoke

# any config value can be overridden on the command line, e.g. the lmax comparison:
.venv/bin/nequip-train -cp $PWD/nequip-md17 -cn ethanol.yaml training_module.model.l_max=2 hydra.run.dir=nequip-md17/outputs/ethanol-lmax2
```

## Model and data at a glance

- **Model:** lmax 1 with parity, 3 layers, 32 channels, cutoff 4 Å, 8 Bessel functions with a p = 6 polynomial cutoff, radial MLP 1×64, float32.
- **Data-dependent parameters:** average neighbour count, per-atom energy shift and force-RMS scale, all measured **per species** from the training split.
- **Training:** Adam (lr 0.01) with EMA weights, ReduceLROnPlateau (×0.6, patience 10), loss = per-atom energy MSE + force MSE.
- **Units:** as in the dataset, kcal/mol and kcal/mol/Å. This datamodule doesn't convert to eV, so pass `energy_units_to_eV=0.0433641` to the ASE calculator for MD.

## Gotchas found while setting this up

- **Loading a model in Python:** call `nequip.utils.global_state.set_global_state()` first, or you get "global state must be initialized before building models".
- **Checkpoint format:** `ModelFromCheckpoint(path)` returns a `ModuleDict`; the model is under `["sole_model"]`. Call `.eval()` before building `NequIPCalculator`.
- **ASE calculator location:** it lives at `nequip.integrations.ase.NequIPCalculator`. The old `nequip.ase` path still works but is deprecated.
- **GPU:** works because pyproject pins torch to the CUDA 12.6 build (the Windows driver only supports CUDA 12.5, and 12.x builds run on it).
