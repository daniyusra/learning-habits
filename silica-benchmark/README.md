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
