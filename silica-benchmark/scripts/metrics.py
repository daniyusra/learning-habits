"""Per-family error metrics shared by every predictor (numpy + ASE only, so it runs in any env)."""
import json
import time
from pathlib import Path

import numpy as np

EV_A3_TO_GPA = 160.2177
FAMILY_GROUP = {"liquid": "liquid", "quench": "quench", "amorphous": "amorphous"}  # everything else is a crystal


def group_of(family):
    return FAMILY_GROUP.get(family, "crystal")


def _full3x3(s):
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size == 9:
        return s.reshape(3, 3)
    xx, yy, zz, yz, xz, xy = s
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def score(test, pred_energy, pred_forces, pred_stress=None):
    """test: list of Atoms with REF_energy / REF_forces / REF_stress. Returns a JSON-able dict.

    Errors are reported per group (crystal / liquid / quench / amorphous) and per crystal phase:
    force MAE and RMSE over Cartesian components (eV/Å), energy MAE per atom (meV/atom),
    pressure MAE (GPa).
    """
    rows = []
    for k, a in enumerate(test):
        n = len(a)
        df = np.asarray(pred_forces[k]) - a.arrays["REF_forces"]
        row = {"orig_index": int(a.info["orig_index"]), "family": a.info["family"], "group": group_of(a.info["family"]),
               "natoms": n, "f_abs_sum": float(np.abs(df).sum()), "f_sq_sum": float((df ** 2).sum()), "f_count": int(df.size),
               "e_err_mev_atom": float((pred_energy[k] - a.info["REF_energy"]) / n * 1000)}
        if pred_stress is not None:
            p_ref = -np.trace(_full3x3(a.info["REF_stress"])) / 3 * EV_A3_TO_GPA
            p_pred = -np.trace(_full3x3(pred_stress[k])) / 3 * EV_A3_TO_GPA
            row["p_err_gpa"] = float(p_pred - p_ref)
        rows.append(row)

    def agg(sel):
        if not sel:
            return None
        fc = sum(r["f_count"] for r in sel)
        out = {"n_structures": len(sel), "n_atoms": sum(r["natoms"] for r in sel),
               "force_mae": sum(r["f_abs_sum"] for r in sel) / fc,
               "force_rmse": float(np.sqrt(sum(r["f_sq_sum"] for r in sel) / fc)),
               "energy_mae_mev_atom": float(np.mean([abs(r["e_err_mev_atom"]) for r in sel]))}
        if "p_err_gpa" in sel[0]:
            out["pressure_mae_gpa"] = float(np.mean([abs(r["p_err_gpa"]) for r in sel]))
        return out

    result = {"all": agg(rows)}
    for g in ["crystal", "liquid", "quench", "amorphous"]:
        result[g] = agg([r for r in rows if r["group"] == g])
    result["by_phase"] = {f: agg([r for r in rows if r["family"] == f])
                          for f in sorted({r["family"] for r in rows if r["group"] == "crystal"})}
    result["per_structure"] = [{k: r[k] for k in ("orig_index", "family", "e_err_mev_atom", "p_err_gpa") if k in r}
                               | {"force_mae": r["f_abs_sum"] / r["f_count"]} for r in rows]
    return result


def fit_element_shift(atoms_list, pred_energy, elements=(8, 14)):
    """Least-squares per-element energy shift so that pred + shift ≈ REF (the 'lifted landscape' correction).

    Silica structures are all SiO2, so only E_Si + 2 E_O is pinned down unless pure Si is present;
    lstsq then returns the minimum-norm split, which is fine for SiO2 and honest about Si-diamond.
    """
    X = np.array([[np.sum(a.numbers == z) for z in elements] for a in atoms_list], dtype=float)
    y = np.array([a.info["REF_energy"] - e for a, e in zip(atoms_list, pred_energy)])
    shift, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {int(z): float(s) for z, s in zip(elements, shift)}


def apply_shift(atoms_list, energies, shift):
    return [e + sum(shift[z] * np.sum(a.numbers == z) for z in shift) for a, e in zip(atoms_list, energies)]


def save(path, approach, seed, n_train, metrics, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"approach": approach, "seed": seed, "n_train": n_train, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec.update(extra or {})
    rec["metrics"] = metrics
    path.write_text(json.dumps(rec, indent=1))
    a = metrics["all"]
    print(f"[{approach} s{seed} n{n_train}] force MAE {a['force_mae']:.3f} eV/Å | energy MAE {a['energy_mae_mev_atom']:.1f} meV/atom"
          + (f" | pressure MAE {a['pressure_mae_gpa']:.2f} GPa" if "pressure_mae_gpa" in a else "") + f" -> {path}")
