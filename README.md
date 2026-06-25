# PCU: Perturbation-Calibrated Uncertainty for Unsupervised Anomaly Detection

PCU is a representation-learning framework for unsupervised anomaly detection.  
It learns uncertainty from controlled perturbations rather than reconstruction error, density estimation, or distance-based scoring.

The main idea is to train an encoder so that its latent response to a ladder of perturbations is calibrated. During training, PCU encourages latent displacement to grow consistently with perturbation strength using ranked sensitivity, scale-aware prediction, and variance-covariance regularization. At inference time, the anomaly score combines three complementary uncertainty cues:

1. Scale response: how strongly the sample appears perturbed according to the learned scale head.
2. Prototype deviation: distance from the nominal embedding prototype.
3. Tiny-noise sensitivity: local instability under infinitesimal perturbations.

This repository provides a clean implementation of PCU and a runner for the 10 representative ADBench datasets used in the paper.

---

## ✨ Highlights

- Label-free unsupervised anomaly detection.
- One-class training protocol: only nominal samples are used during training.
- Perturbation-calibrated representation learning.
- Hybrid anomaly score combining global displacement, learned perturbation sensitivity, and local stability.
- Includes 10 representative ADBench datasets in `.npz` format.
- Reports AUROC mean ± standard deviation across multiple random seeds.
- Saves per-seed JSON logs and anomaly-score arrays.

---

## Repository Structure

```text
PCU/
├── PCU.py
├── run_pcu_adbench10.py
├── _global.json
├── README.md
├── .gitignore
└── ADBench_10_Datasets/
    ├── WDBC.npz
    ├── Ionosphere.npz
    ├── Wilt.npz
    ├── annthyroid.npz
    ├── cover.npz
    ├── http.npz
    ├── letter.npz
    ├── optdigits.npz
    ├── pendigits.npz
    └── thyroid.npz
````

---

## Requirements

Python 3.10 or later is recommended.

Main dependencies:

```bash
pip install numpy scikit-learn torch
```

If you want to run PCU on GPU, install a CUDA-enabled PyTorch version compatible with your system.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/M-Allaoui/PCU.git
cd PCU
```

Create and activate an environment:

```bash
conda create -n pcu python=3.10
conda activate pcu
```

Install dependencies:

```bash
pip install numpy scikit-learn torch
```

Alternatively, using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux or macOS
# .venv\Scripts\activate       # Windows

pip install numpy scikit-learn torch
```

---

## Datasets

The repository includes 10 representative ADBench datasets under:

```text
ADBench_10_Datasets/
```

Each dataset is stored as an `.npz` file and loaded automatically by the runner script.

The included datasets are:

* WDBC
* Pendigits
* HTTP
* Letter
* Ionosphere
* Wilt
* Cover
* OptDigits
* Thyroid
* Annthyroid

---

## Running PCU

### Run on all 10 datasets

```bash
python run_pcu_adbench10.py \
  --data_dir ADBench_10_Datasets \
  --config_json_default _global.json \
  --seeds 3 \
  --device cuda
```

For CPU:

```bash
python run_pcu_adbench10.py \
  --data_dir ADBench_10_Datasets \
  --config_json_default _global.json \
  --seeds 3 \
  --device cpu
```

### Run on one dataset

Example with WDBC:

```bash
python run_pcu_adbench10.py \
  --data_dir ADBench_10_Datasets \
  --dataset WDBC \
  --config_json_default _global.json \
  --seeds 3 \
  --device cuda
```

You can also pass the dataset filename directly:

```bash
python run_pcu_adbench10.py \
  --data_dir ADBench_10_Datasets \
  --dataset WDBC.npz \
  --config_json_default _global.json \
  --seeds 3
```

---

## Outputs

The script prints AUROC mean ± standard deviation across seeds.

Example console output:

```text
[summary] WDBC: AUROC 1.0000 ± 0.0000 | Time 39.79s ± 0.85s over 3 seed(s)
```

By default, results are saved under:

```text
pcu_results/
```

The output folder contains:

```text
pcu_results/
├── <dataset>_seed<seed>.json
└── scores/
    └── <dataset>_seed<seed>_scores.npy
```

The JSON files store per-seed evaluation statistics, and the score arrays store the anomaly scores produced by PCU.

---

## Reproducibility

For stable and comparable results:

* Use the same datasets and preprocessing.
* Keep the one-class protocol unchanged: training uses only nominal samples.
* Run multiple random seeds and report mean ± standard deviation.
* Use the same configuration file `_global.json`.
* Use the same hardware setting when comparing runtime.

The default configuration is provided in:

```text
_global.json
```

---

## Method Summary

Given nominal training data, PCU trains an encoder using a ladder of controlled perturbations. The objective combines:

1. Ranked-sensitivity loss, which encourages latent displacement to increase with perturbation magnitude.
2. Scale-regression loss, which trains a scale head to estimate the injected perturbation magnitude.
3. Variance-covariance regularization, which stabilizes the latent geometry and prevents collapse.
4. EMA prototype tracking, which provides a nominal reference point in representation space.

At test time, PCU computes a final anomaly score from standardized scale response, prototype deviation, and tiny-noise sensitivity.

---

## Notes on Perturbations

The current implementation uses additive Gaussian perturbations, which are most appropriate for standardized continuous or approximately continuous features.

For mixed, binary, ordinal, or categorical tabular data, feature-type-aware perturbation ladders may be more appropriate. This repository provides the Gaussian instantiation used in the main experiments.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{allaoui2026pcu,
  title     = {{PCU}: Perturbation-Calibrated Uncertainty for Unsupervised Anomaly Detection},
  author    = {Mebarka Allaoui and Rachid Hedjam and Mohand Sa\"id Allili and Guoqiang Zhong},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence},
  year      = {2026}
}
```
```
