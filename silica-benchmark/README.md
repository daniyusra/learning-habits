# Silica learning-curve benchmark (stage 4)

Question: for SiO2 at SCAN accuracy, how does error fall with training-set size for
SchNet (from scratch), MACE (from scratch) and MACE-MP-0 (fine-tuned)?

## Data
Erhard, Rohrer, Albe, Deringer, npj Comput. Mater. 2022. Zenodo record 6353684 (CC-BY-4.0):
https://zenodo.org/records/6353684 — `database/dataset.scan.2.xyz`, 3,074 structures, VASP + SCAN.

To reproduce the download:

```bash
mkdir -p silica-benchmark/data/zenodo-6353684 && cd silica-benchmark/data/zenodo-6353684
curl -L -o sio2_potential_data.zip https://zenodo.org/api/records/6353684/files/sio2_potential_data.zip/content
unzip sio2_potential_data.zip 'sio2_potential_data/database/*' 'sio2_potential_data/README'
```

Then run `silica-data.ipynb` (repo root) top to bottom.

## Files
- tracked: `split_v1.json`, which holds the frozen test set, the training pool, the learning-curve sizes and the 3 seed orderings (indices into the original file), plus the isolated-atom energies
- ignored (`data/`, regenerable): the Zenodo download, and `data/splits/{test,pool,isolated_atoms}.xyz` with labels stored as `REF_energy`, `REF_forces` and `REF_stress`

## Data gotchas
- Train on `stress` (it follows the ASE sign convention). Ignore `virials`, which has the opposite sign from the usual convention.
- Melt-quench frames come in liquid/quench/amorphous triplets from one run, so the split groups them by cell volume.
- Dimers, small clusters and isolated atoms are kept out of the test set and the learning-curve subsets. The isolated atoms are only used for reference energies.
- MACE-MP-0 was trained on PBE, while this data is SCAN, so compare untouched MACE-MP-0 on forces, not raw energies.

## MACE environment (separate from the main `.venv`)
mace-torch 0.3.16 pins `e3nn==0.4.4`, while nequip (main env) needs e3nn 0.6, so MACE gets its own venv (git-ignored):

```bash
uv venv silica-benchmark/.venv-mace --python 3.11
VIRTUAL_ENV=silica-benchmark/.venv-mace uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=silica-benchmark/.venv-mace uv pip install mace-torch==0.3.16 ipykernel
```

Foundation model: `mace_mp(model="medium")`, the original MACE-MP-0 (MIT license, 4.7M params, cutoff 6 Å), cached in
`~/.cache/mace/`. Not used: `mace-matpes-r2scan-0` (closer to SCAN, but Academic Software License).

## SchNet environment
SchNetPack 2.2 needs Python 3.12, and on 3.11 it would downgrade torch, so it gets a third venv (git-ignored):

```bash
uv venv silica-benchmark/.venv-schnet --python 3.12
VIRTUAL_ENV=silica-benchmark/.venv-schnet uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=silica-benchmark/.venv-schnet uv pip install schnetpack==2.2.0
```

## Step 1: learning curve at N = 10, 30, 100, 300 (3 seeds)
Scripts in `scripts/`, run from the repo root:
- `make_subsets.py` (any env): writes the fixed 50-structure validation set and the training subsets to `data/lc/`, and the plan to `lc_plan_v1.json`
- `predict_mace.py` (mace env): the zero-shot MACE-MP-0 point. Energies are shifted per element using the validation set.
- `train_mace.py {ft|scratch} SEED N` (mace env) and `train_schnet.py SEED N` (schnet env): train, then score on `test.xyz`
- `metrics.py`: per-family force, energy and pressure errors, shared by every approach
- `run_queue.sh`: the whole grid, resumable, launched detached. Output goes to `results/<approach>/s<seed>_n<N>.json` (tracked) and `runs/` (ignored).

Protocol, the same for all approaches unless noted:
- same validation set, batch size 2 (the 4 GB GPU limit for MACE), float32
- max epochs 200 / 100 / 50 / 30 for N = 10 / 30 / 100 / 300 (about 1,000 to 4,500 gradient steps), with early stopping and the best validation checkpoint kept
- **MACE fine-tune**: naive (single head, no replay of MP data), lr 1e-3. E0s are MACE-MP-0's plus a per-element shift fitted on the N training structures.
- **MACE from scratch**: MACE-MP-0's medium architecture (L=1, 128 channels, correlation 3, 2 layers, r_max 6), E0s from the SCAN isolated atoms, lr 1e-2. It has 0.66M parameters against 4.7M, only because per-element weights cover 2 elements instead of 89.
- **SchNet**: 128 features, 6 interactions, 50 Gaussians, cosine cutoff 6 Å, lr 5e-4. It validates every epoch, with patience scaled to match MACE.
- Loss weights: MACE energy/atom 1, forces 10, stress 100. SchNet total energy 0.01, forces 1, stress 10. These are reasonable defaults, not tuned per method.
- The equal step budget is deliberate: it is the practical question. A from-scratch model may be undertrained at this budget, so check its validation curves.
