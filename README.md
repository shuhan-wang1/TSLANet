# TSLANet: Rethinking Transformers for Time Series Representation Learning [[Paper](https://arxiv.org/pdf/2404.08472.pdf)] [[Poster](https://icml.cc/media/icml-2024/Slides/34691.pdf)] [[Cite](#citation)]
#### *by: Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu,and Xiaoli Li*

### This work is accepted in ICML 2024!

## Abstract
<p align="center">
<img src="misc/TSLANet.png" width="600" class="center">
</p>

Time series data, characterized by its intrinsic long and short-range dependencies, poses a unique challenge across analytical applications. While Transformer-based models excel at capturing long-range dependencies, they face limitations in noise sensitivity, computational efficiency, and overfitting with smaller datasets. In response, we introduce a novel <b>T</b>ime <b>S</b>eries <b>L</b>ightweight <b>A</b>daptive <b>Net</b>work (<b>TSLANet</b>), as a universal convolutional model for diverse time series tasks. Specifically, we propose an Adaptive Spectral Block, harnessing Fourier analysis to enhance feature representation and to capture both long-term and short-term interactions while mitigating noise via adaptive thresholding. Additionally, we introduce an Interactive Convolution Block and leverage self-supervised learning to refine the capacity of TSLANet for decoding complex temporal patterns and improve its robustness on different datasets. Our comprehensive experiments demonstrate that TSLANet outperforms state-of-the-art models in various tasks spanning classification, forecasting, and anomaly detection, showcasing its resilience and adaptability across a spectrum of noise levels and data sizes.

---

## Probabilistic Forecasting Extension

This repository includes a **probabilistic extension** of TSLANet for uncertainty-aware time series forecasting. The extension replaces deterministic point predictions with full Gaussian predictive distributions, enabling decomposition of predictive uncertainty into **aleatoric** (data noise) and **epistemic** (model uncertainty) components.

### Key Features

- **Heteroscedastic Gaussian Likelihood** — The model learns input-dependent mean and variance via dual output heads, trained with negative log-likelihood (NLL) loss
- **MC Dropout Inference** — Leverages the existing dropout in TSLANet's Interactive Convolution Blocks to perform approximate Bayesian inference at test time
- **Deep Ensembles** — Trains multiple models with different random seeds and combines predictions as a mixture of Gaussians
- **Uncertainty Decomposition** — Separates total predictive variance into:
  - Aleatoric variance (mean of predicted variances across stochastic passes)
  - Epistemic variance (variance of predicted means across stochastic passes)
- **Proper Scoring Rules** — Evaluation with NLL and CRPS (Continuous Ranked Probability Score), plus calibration analysis and PIT histograms
- **LSTM Baseline** — Channel-independent Gaussian LSTM with identical normalization and output interface for fair comparison

### Mathematical Formulation

The probabilistic model predicts Gaussian parameters $\mu_t(\mathbf{x}), \sigma_t^2(\mathbf{x})$ and is trained by minimizing:

$$\mathcal{L}(\theta) = \frac{1}{T}\sum_{t=1}^{T}\left[\frac{1}{2}\log\sigma_t^2 + \frac{(y_t - \mu_t)^2}{2\sigma_t^2}\right]$$

Under MC Dropout with $K$ stochastic forward passes, total variance decomposes as:

$$\text{Var}[y] = \underbrace{\frac{1}{K}\sum_{k=1}^{K}\sigma^2_k(\mathbf{x})}_{\text{aleatoric}} + \underbrace{\frac{1}{K}\sum_{k=1}^{K}\left(\mu_k(\mathbf{x}) - \bar{\mu}\right)^2}_{\text{epistemic}}$$

### File Structure

```
Forecasting/probabilistic/
├── prob_model.py            # ProbabilisticTSLANet (dual-head mu + log_var)
├── prob_losses.py           # GaussianNLLLoss, CRPSLoss
├── prob_training.py         # Lightning modules for pretraining and fine-tuning
├── prob_inference.py        # MC Dropout and Deep Ensemble inference
├── prob_metrics.py          # NLL, CRPS, calibration, sharpness, PIT
├── prob_visualization.py    # Prediction intervals, uncertainty decomposition plots
├── baseline_lstm.py         # LSTM + Gaussian baseline
├── run_probabilistic.py     # Unified entry point for all experiments
└── scripts/
    └── weather_prob.sh      # Full ablation suite (6 configs x 4 horizons)
```

### Quick Start

```bash
cd Forecasting/probabilistic

# Probabilistic TSLANet with MC Dropout (K=50) on Weather dataset
python run_probabilistic.py \
    --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 \
    --probabilistic True --mc_dropout True --mc_samples 50

# Deterministic baseline (MSE loss, for ablation)
python run_probabilistic.py \
    --data custom --root_path ../data/weather --data_path weather.csv \
    --probabilistic False

# Deep Ensemble (5 members)
python run_probabilistic.py \
    --data custom --root_path ../data/weather --data_path weather.csv \
    --probabilistic True --deep_ensemble True --ensemble_size 5

# LSTM baseline with MC Dropout
python run_probabilistic.py \
    --data custom --root_path ../data/weather --data_path weather.csv \
    --model_type lstm --probabilistic True --mc_dropout True

# Run full ablation suite
bash scripts/weather_prob.sh
```

### Ablation Experiments

| ID | Model | Loss | MC Dropout | Ensemble | Purpose |
|----|-------|------|------------|----------|---------|
| A1 | TSLANet | MSE | No | No | Deterministic baseline |
| A2 | TSLANet | NLL | No | No | Aleatoric uncertainty only |
| A3 | TSLANet | NLL | Yes (K=50) | No | Aleatoric + epistemic |
| A4 | TSLANet | NLL | No | Yes (M=5) | Deep Ensemble |
| A5 | LSTM | NLL | Yes (K=50) | No | Baseline comparison |
| A6 | LSTM | MSE | No | No | LSTM deterministic baseline |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **NLL** | Negative log-likelihood (proper scoring rule) |
| **CRPS** | Continuous Ranked Probability Score (proper scoring rule) |
| **Calibration** | Observed vs nominal coverage for prediction intervals |
| **Sharpness** | Average width of 90% prediction intervals |
| **MSE / MAE / RMSE** | Standard point prediction metrics on the mean |
| **PIT Histogram** | Probability Integral Transform uniformity check |

### Generated Visualizations

The evaluation pipeline produces:
1. **Prediction interval plots** — Time series with shaded 50/90/95% prediction bands
2. **Uncertainty decomposition** — Stacked area chart of epistemic vs aleatoric variance over forecast horizon
3. **Calibration diagram** — Observed coverage vs nominal coverage (diagonal = perfect)
4. **PIT histogram** — Should be uniform for well-calibrated models
5. **Model comparison** — Bar chart of all metrics across configurations

---

## Datasets
### Forecasting
Forecasting and AD datasets are downloaded from TimesNet https://github.com/thuml/Time-Series-Library

### Classification
- UCR and UEA classification datasets are available at https://www.timeseriesclassification.com
- Sleep-EDF and UCIHAR datasets are from https://github.com/emadeldeen24/TS-TCC
- For any other dataset, to convert to `.pt` format, follow the preprocessing steps here https://github.com/emadeldeen24/TS-TCC/tree/main/data_preprocessing



## Citation
If you found this work useful for you, please consider citing it.
```
@inproceedings{tslanet,
  title     = {TSLANet: Rethinking Transformers for Time Series Representation Learning},
  author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Li, Xiaoli},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

## Acknowledgements
The codes in this repository are inspired by the following:

- GFNet https://github.com/raoyongming/GFNet
- Masking task is from PatchTST https://github.com/yuqinie98/PatchTST
- Forecasting and AD datasets are downloaded from TimesNet https://github.com/thuml/Time-Series-Library
- MC Dropout as approximate Bayesian inference: Gal & Ghahramani, 2016
- Deep Ensembles: Lakshminarayanan et al., 2017
- CRPS evaluation: Gneiting & Raftery, 2007
