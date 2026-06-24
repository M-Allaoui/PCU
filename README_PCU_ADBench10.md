# PCU – Evaluation on `ADBench_10_Datasets`

This folder contains a **minimal** runner script to evaluate **PCU** on the 10 datasets stored as `*.npz` under `ADBench_10_Datasets/`.

The script prints **AUROC mean ± std across seeds for each dataset** (e.g., seeds `0..2`), and saves per-seed JSON logs + score arrays.

---

## Expected project layout

```
PCU/
  PCU.py
  _global.json
  run_pcu_adbench10.py
  README_PCU_ADBench10.md
  Figures/
  pcu_results/
  ADBench_10_Datasets/
```

---

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install numpy scikit-learn torch
```

Notes:
- If you want to run on GPU, install a CUDA-enabled PyTorch build appropriate to your system.
- `PCU.py` / `PCUConfig` are imported from your project.

---

## Run commands

### 1) Run on **all** datasets (10 datasets) with **3 seeds** (0,1,2)

```bash
python run_pcu_adbench10.py --data_dir ADBench_10_Datasets --config_json_default _global.json --seeds 3 --device cuda
```

### 2) Run on **one** dataset only (example: `WDBC`) with **3 seeds**

```bash
python run_pcu_adbench10.py --data_dir ADBench_10_Datasets --dataset WDBC --config_json_default _global.json --seeds 3 --device cuda
```

You can also pass a filename:

```bash
python run_pcu_adbench10.py --data_dir ADBench_10_Datasets --dataset WDBC.npz --config_json_default _global.json --seeds 3
```

---

## Outputs

Default output directory: `results/`

- Per-seed JSON:
  - `pcu_results/<dataset>_seed<seed>.json`
- Per-seed scores:
  - `pcu_results/scores/<dataset>_seed<seed>_scores.npy`

The console output includes a per-dataset summary:

```
[summary] WDBC: AUROC 1.0000 ± 0.0000 | Time 39.79s ± 0.85s over 3 seed(s)
```
