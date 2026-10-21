# Stage 3 Handoff: Equivariance with e3nn

Session date: 2026-10-08. Notebook: `e3nn-equivariance.ipynb` (13 sections, runs top to bottom).

## What was built
- **Sections 1–4:** Irreps, Wigner-D, parity, and the CG tensor product checked against plain dot and cross products.
- **Sections 5–8:** a hand-built e3nn conv on a jittered SiO4 cluster: radius graph, SH edge embedding, radial MLP, TP, neighbour sum, then gate and self-interaction across two layers (`TinyNet`).
- **Sections 9–13:** the checkpoint tests:
  - rotation, using a per-irrep Wigner-D
  - translation
  - parity (det = -1)
  - float32 vs float64
  - energy readout + autograd forces: forces are `1o`, ΣF = 0, symmetric Hessian vs a direct `1o` head
- **Four broken baselines,** each caught by the right test:
  - MLP(edge_vec) weights
  - max aggregation
  - ReLU on vectors
  - raw-xyz MLP / `pos` fed in as a `1o` feature

## Key takeaways (including corrections to earlier notes)
1. **Parity selection rule:** on top of |l1-l2| ≤ l ≤ l1+l2, output parity = p1·p2. So `1o⊗1o → 0e+1e+2e` (dot, cross, traceless part) and never `0o`. A pseudoscalar takes two steps: a·(b×c).
2. **Weights:** one per (in-channel, out-channel) pair *per irrep path* ('uvw' mode), not one per (l1,l2,l) path. The cost scales with multiplicity².
3. **Anything that multiplies a steerable feature as a scalar must be invariant `0e`.** That covers radial weights, gates, and attention coefficients.
4. **Each symmetry needs its own test:**
   - `pos` as `1o` passes rotation but fails translation.
   - Vectors mislabelled `1e` pass rotation but fail parity.
5. **Reading an error:** look at the exponent relative to the output scale.
   - It scales with eps (f32 ~1e-6, f64 ~1e-15): round-off.
   - It stays the same across dtypes: a bug.
   - It sits in between: bisect, *including the test harness*.
6. **Forces = -∂E/∂x** are automatically `1o`, sum to zero, and are conservative. A direct `1o` head is equivariant but **not** conservative, which means energy drift in MD.
7. **TFN** covers SE(3): rotations, translations and permutations, but not reflections. Parity tracking came with e3nn and NequIP.

## Environment gotchas
- **Running code:** use `.venv/bin/python` (there is no system `python`). e3nn 0.6.0 was added via `uv add`.
- **CPU only:** torch 2.14+cu130 can't use CUDA because the WSL NVIDIA driver (12.5) is too old.
- **Missing packages:** `torch_cluster` and `torch_scatter` are not installed, so the notebook uses a `cdist` radius graph and `index_add`.
- **Wigner-D precision:** `D_from_matrix` uses `torch.matrix_exp`. In float64 on a single matrix it is only ~2e-10 accurate for small angles, compared with ~1e-14 for a batch of 2 or more. The workaround is `D_of(irreps, R) = irreps.D_from_matrix(torch.stack([R, I]))[0]`. Only verified on torch 2.14 CPU, and not reported upstream.
- **l=1 basis:** e3nn 0.6.0 orders it (x, y, z), so `Irreps("1o").D_from_matrix(R) == R`.
- **SH normalization:** `spherical_harmonics` defaults to normalization='integral'; the tutorials pass 'component'.

## Open decisions
- **GPU for NequIP training:** update the WSL driver, use a cloud GPU, or train a tiny model on CPU.
- **Optional:** a minimal reproduction of the `matrix_exp` issue for a PyTorch bug report.

## Next session plan
1. **NequIP paper** (arXiv:2101.03164), mapping each block onto this notebook. The new pieces are the Bessel radial basis, residuals, avg-neighbour normalization, and the choice of lmax.
2. **Install and train NequIP 0.19.1** (requires e3nn >=0.6,<0.7; uses Hydra + Lightning configs, not the 2021-era YAML) on a small MD17 subset.
3. **Rerun the notebook's checks on the trained model:** force equivariance and conservativeness.

**Deferred topics:**
- e3nn Transformer tutorial (connects to the GAT)
- periodic boundary conditions (needed for crystalline/amorphous SiO2)
- `0o` / chirality for α-quartz handedness

## Warm-up quiz for next session
1. Which test catches forces mislabelled `1e`, and why can't a rotation test catch it?
2. A float64 equivariance error comes out at 3e-8. What do you do first?
3. Why does max aggregation keep the `0e` outputs equivariant but break l > 0?
