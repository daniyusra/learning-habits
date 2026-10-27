"""MACE predictions + the zero-shot MACE-MP-0 baseline. Runs in silica-benchmark/.venv-mace.

Zero-shot (the N = 0 point), from the repo root:
    silica-benchmark/.venv-mace/bin/python silica-benchmark/scripts/predict_mace.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402

ROOT = Path("silica-benchmark")


def load_calc(model, dtype="float32"):
    """model: a foundation-model name ('medium') or a path to a trained .model file."""
    if Path(str(model)).is_file():
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=str(model), device="cuda", default_dtype=dtype)
    from mace.calculators import mace_mp
    return mace_mp(model=model, device="cuda", default_dtype=dtype)


def predict(calc, atoms_list):
    E, F, S = [], [], []
    for a in atoms_list:
        b = a.copy()
        b.calc = calc
        E.append(b.get_potential_energy())
        F.append(b.get_forces())
        S.append(b.get_stress(voigt=False))
    return E, F, S


def main():
    test = read(ROOT / "data/splits/test.xyz", index=":")
    val = read(ROOT / "data/lc/val.xyz", index=":")
    calc = load_calc("medium", dtype="float64")
    Ev, _, _ = predict(calc, val)
    shift = metrics.fit_element_shift(val, Ev)
    E, F, S = predict(calc, test)
    raw = metrics.score(test, E, F, S)
    m = metrics.score(test, metrics.apply_shift(test, E, shift), F, S)
    metrics.save(ROOT / "results/mace_mp0_zeroshot/s0_n0.json", "mace_mp0_zeroshot", 0, 0, m,
                 {"model": "mace_mp medium (MACE-MP-0, 2023-12-03)", "dtype": "float64",
                  "energy_shift_eV": shift, "energy_shift_fitted_on": "lc val set (50 structures)",
                  "energy_mae_unshifted_mev_atom": raw["all"]["energy_mae_mev_atom"]})


if __name__ == "__main__":
    main()
