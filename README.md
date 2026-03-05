# ProbTSLANet: Probabilistic Time Series Forecasting with Uncertainty Decomposition

> **An extension of TSLANet (ICML 2024) with principled uncertainty quantification via Gaussian NLL and MC Dropout.**

**Base model:** TSLANet: Rethinking Transformers for Time Series Representation Learning [[Paper](https://arxiv.org/pdf/2404.08472.pdf)] [[ICML 2024](https://icml.cc/media/icml-2024/Slides/34691.pdf)]
*by: Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, and Xiaoli Li*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Original TSLANet Architecture](#2-original-tslanet-architecture)
3. [Probabilistic Extensions](#3-probabilistic-extensions)
4. [File Structure](#4-file-structure)
5. [Experiment Matrix](#5-experiment-matrix)
6. [Quick Start](#6-quick-start)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Generated Visualizations](#8-generated-visualizations)
9. [Environment](#9-environment)
10. [Citation](#10-citation)

---

## 1. Project Overview

<p align="center">
<img src="misc/TSLANet.png" width="600" class="center">
</p>

This repository extends TSLANet from a **deterministic point forecaster** into a **full probabilistic forecasting system** that quantifies both **aleatoric uncertainty** (irreducible data noise) and **epistemic uncertainty** (reducible model uncertainty). The original TSLANet outputs a single predicted value for each time step; our extension outputs a full predictive distribution, enabling calibrated prediction intervals and principled uncertainty decomposition.

### Key Contributions

1. **Heteroscedastic Gaussian output heads** — the model learns both a mean and a per-timestep, per-variable log-variance, trained with Gaussian NLL loss
2. **MC Dropout for epistemic uncertainty** — multiple stochastic forward passes at test time to estimate model uncertainty without retraining
3. **Uncertainty decomposition** — clean separation into aleatoric (data noise) and epistemic (model uncertainty) components
4. **Proper scoring rules** — evaluation via NLL, CRPS, calibration curves, and sharpness metrics

---

## 2. Original TSLANet Architecture

TSLANet processes multivariate time series through three main components:

1. **Adaptive Spectral Block (ASB):** Applies FFT, learns frequency-domain weights, and uses an energy-based adaptive mask to separate low-frequency trends from high-frequency noise
2. **Interactive Convolutional Block (ICB):** Two parallel 1D convolutions (kernel sizes 1 and 3) with multiplicative gating, capturing both local patterns and cross-feature interactions
3. **Patch-based input:** The input sequence is split into overlapping patches (stride = patch_size / 2), embedded, and processed through stacked TSLANet layers

Instance normalization (RevIN-style) is applied per-sample, per-variable before the backbone, and the output is denormalized back to the original scale.

---

## 3. Probabilistic Extensions

### 3.1 Heteroscedastic Gaussian Output Heads

Instead of a single linear projection to `pred_len`, the probabilistic model uses two heads:

```
mu_head:      Linear(backbone_out_dim → pred_len)   # predicted mean
log_var_head: Linear(backbone_out_dim → pred_len)   # predicted log-variance
```

**Training loss:** Gaussian NLL (without constant term, since it doesn't affect gradients):

$$\mathcal{L} = \frac{1}{2} \mathbb{E}\left[\log \sigma^2 + \frac{(y - \mu)^2}{\sigma^2}\right]$$

**Denormalization:** Since the backbone operates in normalized space, the outputs are mapped back:
- `mu = mu_norm * stdev + mean`
- `log_var = log_var_norm + 2 * log(stdev)`

### 3.2 MC Dropout for Epistemic Uncertainty

At test time, we run `K` stochastic forward passes with dropout enabled (only `nn.Dropout` layers, not `DropPath`):

- **Epistemic variance** = variance of predicted means across K passes
- **Aleatoric variance** = mean of predicted variances across K passes (Gaussian mode) or 0 (deterministic mode)
- **Total variance** = epistemic + aleatoric

### 3.3 Four Experimental Configurations

| Config | Loss | MC Dropout | Aleatoric | Epistemic | Description |
|--------|------|------------|-----------|-----------|-------------|
| **A1** | MSE | No | - | - | Deterministic baseline |
| **A2** | MSE | Yes | - | Yes | Epistemic only via MC Dropout |
| **A3** | Gaussian NLL | No | Yes | - | Aleatoric only via Gaussian heads |
| **A4** | Gaussian NLL | Yes | Yes | Yes | Full uncertainty decomposition |

---

## 4. File Structure

```
Forecasting/
├── train.py            # Trains model, saves weights. Handles pretraining + fine-tuning.
├── test.py             # Loads saved model, produces metrics JSON + plots.
├── model.py            # ProbabilisticTSLANet (single class, supports det + prob modes)
├── losses.py           # GaussianNLLLoss only (~18 lines)
├── inference.py        # 3 functions: deterministic_predict, gaussian_predict, mc_dropout_predict
├── metrics.py          # compute_nll, compute_crps, compute_calibration, compute_all_metrics
├── visualization.py    # 5 plot functions + generate_all_plots
├── data_factory.py     # Data provider factory
├── data_loader.py      # Dataset classes (ETT + Custom), numpy-based StandardScaler
├── timefeatures.py     # Time feature extraction (from GluonTS)
├── utils.py            # random_masking_3D, str2bool, DropPath, trunc_normal_
├── scripts/
│   └── run_all.sh      # Runs all 8 experiments (4 configs × 2 pred_lens)
└── saved_models/       # Created by train.py at runtime
```

### Design Decisions

- **No global args:** Every class receives `args` via constructor — no fragile module-level globals
- **No external ML dependencies:** `DropPath` and `trunc_normal_` are embedded directly (~25 lines) instead of importing from `timm`; `StandardScaler` uses numpy instead of `sklearn`
- **Pure PyTorch training:** No Lightning — plain training loop with manual best-model tracking
- **JSON output:** Results saved as JSON, not Excel

---

## 5. Experiment Matrix

All experiments run on the **Weather dataset** with two prediction horizons:

| Experiment | Config | pred_len | Probabilistic | MC Dropout |
|------------|--------|----------|---------------|------------|
| 1 | A1 | 96 | False | False |
| 2 | A2 | 96 | False | True |
| 3 | A3 | 96 | True | False |
| 4 | A4 | 96 | True | True |
| 5 | A1 | 336 | False | False |
| 6 | A2 | 336 | False | True |
| 7 | A3 | 336 | True | False |
| 8 | A4 | 336 | True | True |

### Default Hyperparameters (CPU-optimized)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seq_len` | 96 | Input sequence length |
| `label_len` | 48 | Start token length |
| `emb_dim` | 32 | Embedding dimension |
| `depth` | 2 | Number of TSLANet layers |
| `patch_size` | 16 | Patch size |
| `dropout` | 0.3 | Dropout rate |
| `mask_ratio` | 0.4 | Pretraining mask ratio |
| `batch_size` | 32 | Batch size |
| `lr` | 1e-4 | Learning rate |
| `pretrain_epochs` | 5 | Self-supervised pretraining epochs |
| `train_epochs` | 15 | Supervised fine-tuning epochs |
| `mc_samples` | 50 | MC Dropout forward passes |

---

## 6. Quick Start

### Prerequisites

```bash
# CPU-only PyTorch environment
pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib
```

Only 2 extra packages beyond PyTorch: `pandas` and `matplotlib`.

### Training

```bash
cd Forecasting

# A1: Deterministic
python train.py --probabilistic False --mc_dropout False --pred_len 96

# A2: Epistemic only (MSE + MC Dropout at test time)
python train.py --probabilistic False --mc_dropout True --pred_len 96

# A3: Aleatoric only (Gaussian NLL, no MC)
python train.py --probabilistic True --mc_dropout False --pred_len 96

# A4: Full uncertainty (Gaussian NLL + MC Dropout)
python train.py --probabilistic True --mc_dropout True --pred_len 96
```

### Testing

```bash
# Test a saved model (auto-detects configuration from config.json)
python test.py --model_dir saved_models/<run_dir>

# Override MC samples
python test.py --model_dir saved_models/<run_dir> --mc_samples 100
```

### Run All 8 Experiments

```bash
cd Forecasting
bash scripts/run_all.sh
```

### Output Structure

```
saved_models/<run_description>/
├── config.json          # All hyperparameters
├── model_weights.pt     # Model state dict
└── results/
    ├── metrics.json     # All evaluation metrics
    └── plots/           # PDF visualizations (probabilistic models only)
        ├── pred_intervals_sample0.pdf
        ├── calibration.pdf
        ├── uncertainty_decomp_var0.pdf
        └── uncertainty_heatmap.pdf
```

---

## 7. Evaluation Metrics

### Point Metrics
- **MSE** — Mean Squared Error
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

### Probabilistic Metrics (A2–A4 only)
- **NLL** — Gaussian Negative Log-Likelihood (with log(2pi) constant)
- **CRPS** — Continuous Ranked Probability Score (analytic Gaussian)
- **Calibration Error** — Average |nominal - observed| coverage across quantiles
- **Sharpness** — Average prediction interval width at 90% and 50% confidence

### Uncertainty Decomposition
- **Mean Epistemic Variance** — Model uncertainty (reducible with more data)
- **Mean Aleatoric Variance** — Data noise (irreducible)
- **Epistemic Fraction** — Proportion of total uncertainty from epistemic source

---

## 8. Generated Visualizations

For probabilistic models (A2–A4), `test.py` generates:

1. **Prediction Intervals** — Shaded 50%/90%/95% prediction intervals with ground truth overlay
2. **Calibration Plot** — Observed vs nominal coverage (well-calibrated models fall on the diagonal)
3. **Uncertainty Decomposition** — Stacked area chart of aleatoric vs epistemic variance across the forecast horizon
4. **Uncertainty Heatmap** — 4-row heatmap showing absolute error, aleatoric, epistemic, and total uncertainty across all variables

All plots are saved as high-resolution PDFs.

---

## 9. Environment

- Python 3.12
- CPU-only PyTorch (installed via `pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu`)
- pandas
- matplotlib

No other dependencies required. The codebase embeds `DropPath` and `trunc_normal_` directly and uses a numpy-based `StandardScaler`.

---

## 10. Citation

If you find this work useful, please cite the original TSLANet paper:

```bibtex
@inproceedings{eldele2024tslanet,
  title={TSLANet: Rethinking Transformers for Time Series Representation Learning},
  author={Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Li, Xiaoli},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

---

## Acknowledgements

- [TSLANet](https://github.com/emadeldeen24/TSLANet) — Original architecture by Eldele et al.
- [GluonTS](https://github.com/awslabs/gluonts) — Time feature extraction utilities
- COMP0197 Applied Deep Learning, UCL — Coursework context
