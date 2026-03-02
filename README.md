# ProbTSLANet: Probabilistic Time Series Forecasting with Uncertainty Decomposition

> **An extension of TSLANet (ICML 2024) with principled uncertainty quantification via Deep Evidential Regression, MC Dropout, and Deep Ensembles.**

**Base model:** TSLANet: Rethinking Transformers for Time Series Representation Learning [[Paper](https://arxiv.org/pdf/2404.08472.pdf)] [[ICML 2024](https://icml.cc/media/icml-2024/Slides/34691.pdf)]
*by: Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, and Xiaoli Li*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Original TSLANet Architecture](#2-original-tslanet-architecture)
3. [Innovations: From Deterministic to Probabilistic](#3-innovations-from-deterministic-to-probabilistic)
   - [Innovation 1: Heteroscedastic Gaussian Output Heads](#innovation-1-heteroscedastic-gaussian-output-heads)
   - [Innovation 2: Deep Evidential Regression with NIG Priors](#innovation-2-deep-evidential-regression-with-nig-priors)
   - [Innovation 3: Evidence Annealing Schedule](#innovation-3-evidence-annealing-schedule)
   - [Innovation 4: Integrated Coverage Loss](#innovation-4-integrated-coverage-loss)
   - [Innovation 5: MC Dropout as Approximate Bayesian Inference](#innovation-5-mc-dropout-as-approximate-bayesian-inference)
   - [Innovation 6: Deep Ensembles](#innovation-6-deep-ensembles)
   - [Innovation 7: Proper Scoring Rules (CRPS)](#innovation-7-proper-scoring-rules-crps)
   - [Innovation 8: Comprehensive Calibration Analysis](#innovation-8-comprehensive-calibration-analysis)
4. [Summary of Innovation Sources](#4-summary-of-innovation-sources)
5. [File Structure](#5-file-structure)
6. [Complete Experiment Matrix](#6-complete-experiment-matrix)
   - [6.1 Configuration Overview](#61-configuration-overview)
   - [6.2 Progressive Comparison Chains](#62-progressive-comparison-chains)
   - [6.3 Full Hyperparameter Reference](#63-full-hyperparameter-reference)
   - [6.4 Evaluation Metrics per Configuration](#64-evaluation-metrics-per-configuration)
   - [6.5 Generated Visualizations](#65-generated-visualizations)
7. [Quick Start & Running Experiments](#7-quick-start--running-experiments)
   - [7.1 One-Command: Run All Experiments](#71-one-command-run-all-experiments)
   - [7.2 Customization via Environment Variables](#72-customization-via-environment-variables)
   - [7.3 Individual Experiment Commands](#73-individual-experiment-commands)
   - [7.4 Output Structure](#74-output-structure)
8. [COMP0197 Coursework Requirements Q&A](#8-comp0197-coursework-requirements-qa)
9. [Citation](#9-citation)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Project Overview

<p align="center">
<img src="misc/TSLANet.png" width="600" class="center">
</p>

This repository extends TSLANet from a **deterministic point forecaster** into a **full probabilistic forecasting system** that quantifies both **aleatoric uncertainty** (irreducible data noise) and **epistemic uncertainty** (reducible model uncertainty). The original TSLANet outputs a single predicted value $\hat{y}_t$ for each time step; our extension outputs a full predictive distribution $p(y_t | \mathbf{x})$, enabling calibrated prediction intervals and principled uncertainty decomposition.

The key motivating insight comes from ProbFM (Chinta et al., 2025): existing probabilistic time series models either fix distributional families a priori, conflate different sources of uncertainty, or require computationally expensive sampling. We address these limitations by integrating **Deep Evidential Regression (DER)** into the TSLANet backbone, alongside classical approaches (MC Dropout, Deep Ensembles) for comprehensive comparison.

---

## 2. Original TSLANet Architecture

TSLANet (Eldele et al., ICML 2024) is a lightweight convolutional model for time series that combines spectral analysis with interactive convolution. Its core components are:

### 2.1 Patch Embedding with Instance Normalization

Input $\mathbf{x} \in \mathbb{R}^{B \times L \times M}$ is first normalized per-sample (RevIN-style):

$$\bar{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{inst}}}{\boldsymbol{\sigma}_{\text{inst}}}, \quad \boldsymbol{\mu}_{\text{inst}} = \frac{1}{L}\sum_{t=1}^{L} x_t, \quad \boldsymbol{\sigma}_{\text{inst}} = \sqrt{\frac{1}{L}\sum_{t=1}^{L}(x_t - \boldsymbol{\mu}_{\text{inst}})^2 + \epsilon}$$

Then, the sequence is unfolded into overlapping patches with `stride = patch_size / 2` and linearly embedded:

$$\mathbf{X}_{\text{patches}} = \text{Unfold}(\bar{\mathbf{x}}, \text{patch\_size}, \text{stride}), \quad \mathbf{H}^{(0)} = \mathbf{X}_{\text{patches}} \mathbf{W}_{\text{embed}} + \mathbf{b}_{\text{embed}}$$

### 2.2 Adaptive Spectral Block (ASB)

The ASB operates in the frequency domain via the Discrete Fourier Transform. Given input $\mathbf{H} \in \mathbb{R}^{N \times D}$ (where $N$ = num\_patches, $D$ = emb\_dim):

$$\hat{\mathbf{H}} = \text{FFT}(\mathbf{H}), \quad \hat{\mathbf{H}}_{\text{weighted}} = \hat{\mathbf{H}} \odot \mathbf{W}_{\text{freq}}$$

where $\mathbf{W}_{\text{freq}} \in \mathbb{C}^{D}$ is a learnable complex-valued weight. An **adaptive high-frequency mask** $\mathbf{M}$ is computed from the energy spectrum:

$$E_f = \|\hat{\mathbf{H}}_f\|^2, \quad \tilde{E}_f = \frac{E_f}{\text{median}(\{E_f\}) + \epsilon}$$

$$M_f = \mathbb{1}\left[\tilde{E}_f > \tau\right], \quad \hat{\mathbf{H}}_{\text{high}} = (\hat{\mathbf{H}} \odot \mathbf{M}) \odot \mathbf{W}_{\text{high}}$$

where $\tau$ is a learnable threshold parameter and $\mathbf{W}_{\text{high}}$ is a second set of complex weights. The final output is:

$$\mathbf{H}_{\text{ASB}} = \text{IFFT}\left(\hat{\mathbf{H}}_{\text{weighted}} + \hat{\mathbf{H}}_{\text{high}}\right)$$

### 2.3 Interactive Convolution Block (ICB)

The ICB uses two parallel 1D convolution branches with **cross-branch interaction**:

$$\mathbf{z}_1 = \text{Conv}_{1\times 1}(\mathbf{H}), \quad \mathbf{z}_2 = \text{Conv}_{3\times 1}(\mathbf{H})$$

$$\mathbf{o}_1 = \mathbf{z}_1 \odot \text{Dropout}(\text{GELU}(\mathbf{z}_2)), \quad \mathbf{o}_2 = \mathbf{z}_2 \odot \text{Dropout}(\text{GELU}(\mathbf{z}_1))$$

$$\mathbf{H}_{\text{ICB}} = \text{Conv}_{1\times 1}(\mathbf{o}_1 + \mathbf{o}_2)$$

The element-wise cross-multiplication $\mathbf{z}_1 \odot g(\mathbf{z}_2)$ and $\mathbf{z}_2 \odot g(\mathbf{z}_1)$ enables information interaction between the pointwise (global channel mixing) and local (temporal context) branches.

### 2.4 TSLANet Layer and Output

Each TSLANet layer combines ASB and ICB with residual connections:

$$\mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} + \text{DropPath}\left(\text{ICB}\left(\text{LN}\left(\text{ASB}\left(\text{LN}(\mathbf{H}^{(\ell)})\right)\right)\right)\right)$$

The final output is a single linear head, followed by **denormalization**:

$$\hat{\mathbf{y}} = \mathbf{W}_{\text{out}} \cdot \text{flatten}(\mathbf{H}^{(L)}) + \mathbf{b}_{\text{out}}, \quad \hat{\mathbf{y}}_{\text{real}} = \hat{\mathbf{y}} \cdot \boldsymbol{\sigma}_{\text{inst}} + \boldsymbol{\mu}_{\text{inst}}$$

**The original TSLANet is trained with MSE loss:** $\mathcal{L}_{\text{MSE}} = \frac{1}{T}\sum_{t=1}^{T}(y_t - \hat{y}_t)^2$

---

## 3. Innovations: From Deterministic to Probabilistic

### Innovation 1: Heteroscedastic Gaussian Output Heads

**Source:** Nix & Weigend (1994), "Estimating the mean and variance of the target probability distribution"

**What changed:** The original TSLANet has a **single linear output head** producing point predictions. We replace this with **dual output heads** that predict both mean and input-dependent (heteroscedastic) variance:

$$\boldsymbol{\mu}_t(\mathbf{x}) = \mathbf{W}_{\mu} \cdot \text{flatten}(\mathbf{H}^{(L)}) + \mathbf{b}_{\mu}$$

$$\log \sigma_t^2(\mathbf{x}) = \mathbf{W}_{\log\sigma^2} \cdot \text{flatten}(\mathbf{H}^{(L)}) + \mathbf{b}_{\log\sigma^2}$$

The log-variance head is initialized with bias $= -2.0$ (corresponding to $\sigma \approx 0.37$) to prevent NLL explosion at training start. Log-variance is clamped to $[-10, 10]$ for numerical stability.

**Training loss** is the Gaussian Negative Log-Likelihood (NLL):

$$\mathcal{L}_{\text{NLL}}(\theta) = \frac{1}{T}\sum_{t=1}^{T}\left[\frac{1}{2}\log\sigma_t^2 + \frac{(y_t - \mu_t)^2}{2\sigma_t^2}\right]$$

**Denormalization for variance:** Since the model operates on normalized inputs, the predicted variance must be denormalized by scaling quadratically with the instance standard deviation:

$$\log\sigma^2_{\text{real}} = \log\sigma^2_{\text{norm}} + 2\log(\boldsymbol{\sigma}_{\text{inst}})$$

This is because if $\hat{y}_{\text{real}} = \hat{y}_{\text{norm}} \cdot \sigma_{\text{inst}}$, then $\text{Var}[\hat{y}_{\text{real}}] = \sigma^2_{\text{inst}} \cdot \text{Var}[\hat{y}_{\text{norm}}]$.

**Why this matters:** Unlike MSE which treats all predictions as equally confident, the heteroscedastic Gaussian head allows the model to express higher uncertainty for inherently noisier time steps (e.g., volatile market periods, extreme weather events).

---

### Innovation 2: Deep Evidential Regression with NIG Priors

**Source:** Amini et al. (2020), "Deep Evidential Regression" (NeurIPS); adapted for time series following ProbFM (Chinta et al., 2025)

**What changed:** Instead of predicting parameters of a fixed distribution, we predict parameters of a **higher-order distribution over distribution parameters** using a Normal-Inverse-Gamma (NIG) prior. This is the core innovation from ProbFM applied to the TSLANet backbone.

The NIG prior places a conjugate prior over the Gaussian likelihood parameters $(\mu, \sigma^2)$:

$$p(\mu, \sigma^2) = \text{NIG}(\gamma, \nu, \alpha, \beta)$$

which factorizes as:

$$p(\mu \mid \sigma^2) = \mathcal{N}\left(\mu;\; \gamma,\; \frac{\sigma^2}{\nu}\right), \quad p(\sigma^2) = \text{Inverse-Gamma}(\alpha, \beta)$$

where $\gamma \in \mathbb{R}$, $\nu > 0$, $\alpha > 1$, $\beta > 0$ are the four NIG parameters.

**NIGHead architecture** (replacing the single linear layer):

$$\gamma = \mathbf{W}_{\gamma} h + \mathbf{b}_{\gamma} \quad \text{(unconstrained)}$$

$$\nu = \text{Softplus}(\mathbf{W}_{\nu} h + \mathbf{b}_{\nu}) + \epsilon$$

$$\alpha = \text{Softplus}(\mathbf{W}_{\alpha} h + \mathbf{b}_{\alpha}) + 1 + \epsilon$$

$$\beta = \text{Softplus}(\mathbf{W}_{\beta} h + \mathbf{b}_{\beta}) + \epsilon$$

These constraints ensure mathematical validity of the NIG distribution: $\nu > 0$ (positive precision), $\alpha > 1$ (finite variance for the inverse-gamma), $\beta > 0$ (positive scale).

**Closed-form uncertainty decomposition** (no sampling required):

$$\mathbb{E}[y \mid \mathbf{x}] = \gamma$$

$$\text{Aleatoric uncertainty} = \mathbb{E}[\sigma^2 \mid \mathbf{x}] = \frac{\beta}{\alpha - 1}$$

$$\text{Epistemic uncertainty} = \text{Var}[\mu \mid \mathbf{x}] = \frac{\beta}{(\alpha - 1)\nu}$$

$$\text{Total variance} = \frac{\beta(\nu + 1)}{(\alpha - 1)\nu}$$

**NIG Negative Log-Likelihood:**

$$\mathcal{L}_{\text{NLL}}^{\text{NIG}} = \frac{1}{2}\log\frac{\pi}{\nu} - \alpha\log\Omega + \left(\alpha + \frac{1}{2}\right)\log\left[(y - \gamma)^2\nu + \Omega\right] + \log\frac{\Gamma(\alpha)}{\Gamma(\alpha + \frac{1}{2})}$$

where $\Omega = 2\beta(1 + \nu)$.

**Evidence Regularization** (penalizes confident wrong predictions):

$$\mathcal{L}_{\text{reg}} = |y - \gamma| \cdot (2\nu + \alpha)$$

**Total evidential loss:**

$$\mathcal{L}_{\text{EDL}} = \mathcal{L}_{\text{NLL}}^{\text{NIG}} + \lambda_{\text{evd}} \cdot \text{evidence\_scale}(t) \cdot \mathcal{L}_{\text{reg}}$$

**DER denormalization:** Since $\beta$ is a scale parameter for variance, it scales quadratically with the instance standard deviation:

$$\gamma_{\text{real}} = \gamma_{\text{norm}} \cdot \sigma_{\text{inst}} + \mu_{\text{inst}}, \quad \beta_{\text{real}} = \beta_{\text{norm}} \cdot \sigma_{\text{inst}}^2$$

The parameters $\nu$ and $\alpha$ are dimensionless and require no denormalization.

**Key advantage over Gaussian + MC Dropout:** DER provides epistemic-aleatoric decomposition in a **single forward pass** (computational cost = $1\times$), whereas MC Dropout requires $K$ passes (cost = $K\times$) and Deep Ensembles require $M$ separately trained models (cost = $M\times$ training + $M\times$ inference).

---

### Innovation 3: Evidence Annealing Schedule

**Source:** Sensoy, Kaplan & Kandemir (2018), "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS); adapted for regression following ProbFM (Chinta et al., 2025)

**What changed:** We introduce an evidence annealing schedule that linearly scales the regularization strength during training:

$$\text{evidence\_scale}(t) = \min\left(1.0,\; \frac{t}{T_{\text{anneal}}}\right)$$

where $t$ is the current epoch and $T_{\text{anneal}}$ is the annealing period (default 5 epochs).

**Why this matters:** Early in training, when learned representations are still unstable, the model may place excessive confidence in poor predictions. The annealing schedule starts with zero regularization ($\text{evidence\_scale} = 0$), allowing the model to first learn accurate mean predictions, then gradually introduces the evidence penalty to encourage proper calibration. This prevents **evidence collapse** — a common failure mode in Evidential Deep Learning where the model generates overly confident predictions before meaningful patterns are learned.

Unlike Sensoy et al. (2018) who anneal the KL regularization weight, ProbFM and our implementation directly anneal the evidence contribution during optimization, providing more direct control over evidence accumulation.

---

### Innovation 4: Integrated Coverage Loss (Gaussian + Evidential)

**Source:** ProbFM (Chinta et al., 2025) — first integration of coverage loss with Deep Evidential Regression for time series. We extend this to Gaussian predictive distributions, making it applicable to MC Dropout (A3) and Deep Ensemble (A4) configurations.

**What changed:** We add a differentiable **Prediction Interval Coverage Probability (PICP)** loss that directly optimizes calibration:

$$\mathcal{L}_{\text{coverage}} = \left|\text{PICP}_{\text{target}} - \text{PICP}_{\text{actual}}\right|$$

where $\text{PICP}_{\text{target}}$ is the desired coverage probability (e.g., 0.9 for 90% prediction intervals) and:

$$\text{PICP}_{\text{actual}} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[y_i \in [\text{CI}_{\text{lower},i},\; \text{CI}_{\text{upper},i}]\right]$$

Since the indicator function $\mathbb{1}[\cdot]$ is non-differentiable, we use a **sigmoid soft approximation**:

$$\text{soft\_covered}_i = \sigma\left(s \cdot (y_i - \text{CI}_{\text{lower},i})\right) \cdot \sigma\left(s \cdot (\text{CI}_{\text{upper},i} - y_i)\right)$$

where $s$ is a sharpness parameter (default 10.0) controlling the sigmoid steepness.

**Two variants for different predictive distributions:**

| Variant | CI Half-Width | Applicable To |
|---------|---------------|---------------|
| **NIG (Student-t)** | $\text{CI}_{\text{half}} = z \cdot \sqrt{\frac{\beta(\nu + 1)}{\alpha\nu}}$ | DER / Evidential (A7) |
| **Gaussian** | $\text{CI}_{\text{half}} = z \cdot \sigma = z \cdot \exp(0.5 \cdot \log\sigma^2)$ | Gaussian NLL + MC Dropout (A3) / Ensemble (A4) |

where $z = \Phi^{-1}\left(\frac{1 + \text{coverage}}{2}\right)$ is the quantile of the standard normal distribution.

**Combined training objective (works with both modes):**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{base}} + \lambda_{\text{coverage}} \cdot \mathcal{L}_{\text{coverage}}$$

where $\mathcal{L}_{\text{base}}$ is either $\mathcal{L}_{\text{EDL}}$ (for DER) or $\mathcal{L}_{\text{NLL}}^{\text{Gaussian}}$ (for Gaussian mode).

**Why this matters:** The key insight is that the coverage loss objective $|\text{PICP}_{\text{target}} - \text{PICP}_{\text{actual}}|$ is **distribution-agnostic**: it only requires the ability to compute prediction intervals, which any parametric predictive distribution provides. This enables a fair experimental comparison:

- **A3 vs A3+CovLoss:** Isolates the effect of coverage loss on Gaussian + MC Dropout
- **A7 vs A7+CovLoss:** Isolates the effect of coverage loss on DER

Unlike post-hoc calibration methods (e.g., temperature scaling, Platt scaling) that require an additional calibration step after training, the coverage loss directly optimizes prediction interval reliability during training.

---

### Innovation 5: MC Dropout as Approximate Bayesian Inference

**Source:** Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (ICML)

**What changed:** We leverage the existing dropout in TSLANet's ICB blocks to perform approximate Bayesian inference at test time. During inference, dropout remains **enabled** and $K$ stochastic forward passes are performed:

$$\{\boldsymbol{\mu}_k(\mathbf{x}), \log\sigma^2_k(\mathbf{x})\}_{k=1}^{K} \sim q(\boldsymbol{\theta})$$

where $q(\boldsymbol{\theta})$ is the approximate posterior induced by dropout.

**Uncertainty decomposition:**

$$\text{Predictive mean: } \bar{\boldsymbol{\mu}} = \frac{1}{K}\sum_{k=1}^{K}\boldsymbol{\mu}_k(\mathbf{x})$$

$$\text{Aleatoric variance: } \mathbb{V}_{\text{ale}} = \frac{1}{K}\sum_{k=1}^{K}\sigma^2_k(\mathbf{x})$$

$$\text{Epistemic variance: } \mathbb{V}_{\text{epi}} = \frac{1}{K}\sum_{k=1}^{K}\left(\boldsymbol{\mu}_k(\mathbf{x}) - \bar{\boldsymbol{\mu}}\right)^2$$

$$\text{Total variance: } \mathbb{V}_{\text{total}} = \mathbb{V}_{\text{ale}} + \mathbb{V}_{\text{epi}}$$

This is a natural fit for TSLANet since the ICB already contains dropout layers, meaning **no architectural modifications are required** — only the inference procedure changes.

---

### Innovation 6: Deep Ensembles

**Source:** Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (NeurIPS)

**What changed:** We train $M$ independently initialized TSLANet models (different random seeds) and combine their predictions as a mixture of Gaussians:

$$p(y \mid \mathbf{x}) = \frac{1}{M}\sum_{m=1}^{M}\mathcal{N}\left(y;\; \mu_m(\mathbf{x}),\; \sigma^2_m(\mathbf{x})\right)$$

The uncertainty decomposition follows the same law of total variance as MC Dropout:

$$\bar{\boldsymbol{\mu}} = \frac{1}{M}\sum_{m=1}^{M}\boldsymbol{\mu}_m, \quad \mathbb{V}_{\text{ale}} = \frac{1}{M}\sum_{m=1}^{M}\sigma^2_m, \quad \mathbb{V}_{\text{epi}} = \frac{1}{M}\sum_{m=1}^{M}(\boldsymbol{\mu}_m - \bar{\boldsymbol{\mu}})^2$$

---

### Innovation 7: Proper Scoring Rules (CRPS)

**Source:** Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation"

**What changed:** In addition to NLL, we implement the **Continuous Ranked Probability Score (CRPS)** — a strictly proper scoring rule that simultaneously rewards calibration and sharpness. For Gaussian predictive distributions, CRPS has an analytic form:

$$\text{CRPS}(\mu, \sigma, y) = \sigma\left[z\left(2\Phi(z) - 1\right) + 2\phi(z) - \frac{1}{\sqrt{\pi}}\right]$$

where $z = \frac{y - \mu}{\sigma}$, $\Phi(\cdot)$ is the standard normal CDF, and $\phi(\cdot)$ is the standard normal PDF.

**Why this matters:** CRPS operates on the full predictive CDF and measures the integrated squared difference between the predicted CDF and the empirical CDF of the observation. Unlike NLL which can be sensitive to outliers (it penalizes heavily when the observation lies in a low-probability region), CRPS provides a more robust evaluation of distributional forecast quality.

---

### Innovation 8: Comprehensive Calibration Analysis

**Source:** Gneiting, Balabdaoui & Raftery (2007); Dawid (1984)

**What changed:** We implement a comprehensive evaluation suite that goes beyond point prediction metrics:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **NLL** | $-\frac{1}{N}\sum_i \log p(y_i \mid \mu_i, \sigma^2_i)$ | Proper scoring rule |
| **CRPS** | See above | Proper scoring rule |
| **PICP** | $\frac{1}{N}\sum_i \mathbb{1}[y_i \in \text{CI}_i]$ | Interval coverage |
| **Sharpness** | Mean prediction interval width | Interval width |
| **Calibration Error** | $\frac{1}{Q}\sum_q \lvert p_q - \hat{p}_q \rvert$ | Coverage deviation |
| **PIT** | $F(y_i \mid \mu_i, \sigma^2_i)$ | Should be $\sim U(0,1)$ |
| **Uncertainty-Error Correlation** | $\text{Corr}(\sigma_i, \lvert y_i - \mu_i \rvert)$ | Uncertainty quality |

The **Probability Integral Transform (PIT)** computes $u_i = \Phi\left(\frac{y_i - \mu_i}{\sigma_i}\right)$ for each prediction. A well-calibrated model produces PIT values that are uniformly distributed on $[0, 1]$.

---

## 4. Summary of Innovation Sources

| Innovation | Source Paper | Key Idea |
|-----------|-------------|----------|
| Heteroscedastic Gaussian heads | Nix & Weigend (1994) | Learn input-dependent mean and variance |
| Deep Evidential Regression (NIG) | Amini et al. (2020), adapted via ProbFM (Chinta et al., 2025) | Higher-order distribution over parameters; single-pass epistemic-aleatoric decomposition |
| Evidence Annealing | Sensoy et al. (2018), adapted via ProbFM (Chinta et al., 2025) | Gradual evidence regularization to prevent overconfidence |
| Integrated Coverage Loss (NIG) | ProbFM (Chinta et al., 2025) | Differentiable PICP optimization during DER training |
| Gaussian Coverage Loss (ours) | Extended from ProbFM (Chinta et al., 2025) | Coverage loss for Gaussian NLL + MC Dropout / Ensemble |
| NIG parameter constraints (Softplus + offsets) | ProbFM (Chinta et al., 2025) | Ensures NIG validity: $\nu > 0$, $\alpha > 1$, $\beta > 0$ |
| Variance denormalization ($\beta \cdot \sigma_{\text{inst}}^2$) | Our adaptation | Correct scaling of NIG scale parameter after RevIN normalization |
| MC Dropout inference | Gal & Ghahramani (2016) | Approximate Bayesian inference via stochastic forward passes |
| Deep Ensembles | Lakshminarayanan et al. (2017) | Mixture of independently trained models |
| CRPS evaluation | Gneiting & Raftery (2007) | Strictly proper scoring rule for distributional forecasts |
| PIT / Calibration analysis | Dawid (1984); Gneiting et al. (2007) | Probabilistic forecast verification |

---

## 5. File Structure

```
Forecasting/probabilistic/
├── prob_model.py            # ProbabilisticTSLANet (dual-head mu + log_var, or NIGHead)
├── prob_der.py              # NIGHead, EvidentialLoss, CoverageLoss, der_predict()
├── prob_losses.py           # GaussianNLLLoss, CRPSLoss, GaussianCoverageLoss
├── prob_training.py         # Lightning modules for pretraining and fine-tuning
├── prob_inference.py        # MC Dropout and Deep Ensemble inference
├── prob_metrics.py          # NLL, CRPS, calibration, sharpness, PIT
├── prob_visualization.py    # Prediction intervals, heatmaps, calibration, PIT plots
├── baseline_lstm.py         # LSTM + Gaussian / Evidential baseline
├── shared_config.py         # Unified argument parser and config save/load
├── train.py                 # Training entry point (saves models + config)
├── test.py                  # Testing entry point (loads models, runs inference, computes metrics)
├── run_probabilistic.py     # Unified train + test wrapper
└── scripts/
    ├── run_all_experiments.sh  # One-click full experiment suite (10 configs × 4 horizons)
    └── weather_prob.sh        # Legacy ablation suite
```

---

## 6. Complete Experiment Matrix

### 6.1 Configuration Overview

We define **10 experimental configurations** spanning three uncertainty quantification paradigms (Gaussian NLL, MC Dropout, Deep Evidential Regression) across two backbone architectures (TSLANet, LSTM). Each configuration is tested on **4 prediction horizons** ($T \in \{96, 192, 336, 720\}$), yielding **40 total experiments**.

| ID | Backbone | Training Loss | Uncertainty Method | Coverage Loss | Inference | Passes | Decomposition | Purpose |
|----|----------|---------------|--------------------|---------------|-----------|--------|---------------|---------|
| **A1** | TSLANet | MSE | None (deterministic) | No | Single-pass | 1 | None | Point prediction baseline |
| **A2** | TSLANet | Gaussian NLL | Heteroscedastic variance | No | Single-pass | 1 | Aleatoric only | Aleatoric uncertainty baseline |
| **A3** | TSLANet | Gaussian NLL | MC Dropout (K=50) | No | Stochastic | 50 | Aleatoric + Epistemic | Full uncertainty decomposition |
| **A3+** | TSLANet | Gaussian NLL + CovLoss | MC Dropout (K=50) | **Yes** (Gaussian) | Stochastic | 50 | Aleatoric + Epistemic | Calibration-optimized MC Dropout |
| **A4** | TSLANet | Gaussian NLL | Deep Ensemble (M=5) | No | Ensemble | 5 | Aleatoric + Epistemic | Ensemble-based decomposition |
| **A5** | LSTM | Gaussian NLL | MC Dropout (K=50) | No | Stochastic | 50 | Aleatoric + Epistemic | Architecture comparison |
| **A6** | LSTM | MSE | None (deterministic) | No | Single-pass | 1 | None | LSTM point baseline |
| **A7** | TSLANet | Evidential (NIG NLL + reg.) | DER: NIG prior | No | Single-pass | 1 | Aleatoric + Epistemic | Single-pass decomposition |
| **A7+** | TSLANet | Evidential + CovLoss | DER: NIG prior | **Yes** (Student-t) | Single-pass | 1 | Aleatoric + Epistemic | Calibration-optimized DER |
| **A8** | LSTM | Evidential (NIG NLL + reg.) | DER: NIG prior | No | Single-pass | 1 | Aleatoric + Epistemic | DER on LSTM backbone |

> **Notation:** "CovLoss" = Coverage Loss ($\lambda_{\text{cov}} \cdot \mathcal{L}_{\text{coverage}}$); "reg." = evidence regularization ($\lambda_{\text{evd}} \cdot \mathcal{L}_{\text{reg}}$); K = number of MC Dropout forward passes; M = number of ensemble members.

**Key distinctions:**

- **A1 vs A6:** Isolates the backbone effect (TSLANet vs LSTM) with identical deterministic MSE training.
- **A2 vs A3:** Isolates the effect of MC Dropout — both use Gaussian NLL, but A3 adds $K=50$ stochastic passes for epistemic uncertainty.
- **A3 vs A3+:** Isolates the effect of **Gaussian coverage loss** — same model, but A3+ adds $\mathcal{L}_{\text{coverage}}$ during training.
- **A3 vs A7:** Isolates the **uncertainty method** — MC Dropout (50 passes) vs DER (single pass). Both decompose aleatoric + epistemic.
- **A7 vs A7+:** Isolates the effect of **NIG coverage loss** on DER.
- **A3 vs A5, A7 vs A8:** Isolates the **backbone** effect for each uncertainty method.

---

### 6.2 Progressive Comparison Chains

The experiments support two main comparison narratives:

#### Chain 1: TSLANet Uncertainty Progression (Recommended)

```
A1 ──→ A2 ──→ A3 ──→ A3+ ──→ A7 ──→ A7+
 │       │       │       │       │       │
 │       │       │       │       │       └─ + NIG Coverage Loss (best calibration?)
 │       │       │       │       └─ DER: single-pass decomposition (50× speedup)
 │       │       │       └─ + Gaussian Coverage Loss (improved calibration)
 │       │       └─ + MC Dropout (epistemic via K=50 passes)
 │       └─ + Gaussian NLL (learn aleatoric uncertainty)
 └─ Deterministic MSE baseline (no uncertainty)
```

Each step adds exactly **one factor**, enabling clean ablation analysis:

| Transition | Factor Added | Expected Effect |
|-----------|-------------|-----------------|
| A1 → A2 | Gaussian NLL (heteroscedastic $\sigma^2$) | Learns aleatoric uncertainty; NLL, CRPS become available |
| A2 → A3 | MC Dropout ($K=50$ passes) | Epistemic uncertainty appears; better calibration |
| A3 → A3+ | Gaussian Coverage Loss ($\lambda_{\text{cov}}=0.1$) | PICP approaches 90% target; may widen/narrow intervals |
| A3 → A7 | Replace Gaussian+MC with DER (NIG prior) | Single-pass epistemic; 50× inference speedup |
| A7 → A7+ | NIG Coverage Loss ($\lambda_{\text{cov}}=0.1$) | PICP approaches 90% target for DER intervals |

#### Chain 2: Architecture Comparison

```
TSLANet (A3) ←──compare──→ LSTM (A5)      [Gaussian + MC Dropout]
TSLANet (A7) ←──compare──→ LSTM (A8)      [Evidential / DER]
TSLANet (A1) ←──compare──→ LSTM (A6)      [Deterministic baseline]
```

---

### 6.3 Full Hyperparameter Reference

The table below lists every configuration-specific hyperparameter. Shared hyperparameters (applied to **all** configs) are listed first.

**Shared hyperparameters (all experiments):**

| Parameter | Value | CLI Flag |
|-----------|-------|----------|
| Input sequence length | 96 | `--seq_len 96` |
| Prediction horizons | 96, 192, 336, 720 | `--pred_len {96,192,336,720}` |
| Embedding dimension | 64 | `--emb_dim 64` |
| TSLANet depth | 3 layers | `--depth 3` |
| Batch size | 64 | `--batch_size 64` |
| Dropout rate | 0.5 | `--dropout 0.5` |
| Patch size | 64 | `--patch_size 64` |
| Training epochs | 20 | `--train_epochs 20` |
| Pretraining epochs | 10 | `--pretrain_epochs 10` |
| Random seed | 42 | `--seed 42` |
| Features | Multivariate (M) | `--features M` |

**Per-configuration hyperparameters:**

| Parameter | A1 | A2 | A3 | A3+ | A4 | A5 | A6 | A7 | A7+ | A8 |
|-----------|----|----|----|----|----|----|----|----|-----|----|
| `--model_type` | tslanet | tslanet | tslanet | tslanet | tslanet | lstm | lstm | tslanet | tslanet | lstm |
| `--probabilistic` | False | True | True | True | True | True | False | True | True | True |
| `--uncertainty_method` | gaussian | gaussian | gaussian | gaussian | gaussian | gaussian | gaussian | evidential | evidential | evidential |
| `--mc_dropout` | False | False | True | True | False | True | False | False | False | False |
| `--mc_samples` | — | — | 50 | 50 | — | 50 | — | — | — | — |
| `--deep_ensemble` | False | False | False | False | True | False | False | False | False | False |
| `--ensemble_size` | — | — | — | — | 5 | — | — | — | — | — |
| `--use_coverage_loss` | False | False | False | **True** | False | False | False | False | **True** | False |
| `--coverage_target` | — | — | — | 0.9 | — | — | — | — | 0.9 | — |
| `--lambda_coverage` | — | — | — | 0.1 | — | — | — | — | 0.1 | — |
| `--lambda_evd` | — | — | — | — | — | — | — | 0.05 | 0.05 | 0.05 |
| `--anneal_epochs` | — | — | — | — | — | — | — | 5 | 5 | 5 |
| `--lstm_hidden` | — | — | — | — | — | 128 | 128 | — | — | 128 |
| `--lstm_layers` | — | — | — | — | — | 2 | 2 | — | — | 2 |

---

### 6.4 Evaluation Metrics per Configuration

All configurations are evaluated on the same comprehensive metric suite (formulas defined in [Innovation 8](#innovation-8-comprehensive-calibration-analysis)). ✓ = meaningful; ✗ = not applicable.

| Category | Metric | A1 | A2 | A3/A3+ | A4 | A5 | A6 | A7/A7+ | A8 |
|----------|--------|----|----|----|----|----|----|----|-----|
| **Point accuracy** | MSE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| | MAE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| | RMSE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Proper scoring** | NLL | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| | CRPS | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| **Calibration** | PICP | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| | Calibration Error | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| | PIT histogram | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| **Sharpness** | 90% PI width | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| **Uncertainty** | Epistemic fraction | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| | Uncertainty-Error Corr. | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |

---

### 6.5 Generated Visualizations

Each experiment automatically generates the following plots via `prob_visualization.py`:

| # | Plot | Function | Description | Applicable Configs |
|---|------|----------|-------------|--------------------|
| 1 | Prediction Intervals | `plot_prediction_intervals()` | Time series with 50%/90% shaded CI bands | A2, A3, A3+, A4, A5, A7, A7+, A8 |
| 2 | Uncertainty Decomposition | `plot_uncertainty_decomposition()` | Stacked area: aleatoric vs epistemic over time | A3, A3+, A4, A5, A7, A7+, A8 |
| 3 | Calibration Diagram | `plot_calibration()` | Expected vs observed coverage at multiple levels | A2, A3, A3+, A4, A5, A7, A7+, A8 |
| 4 | PIT Histogram | `plot_pit_histogram()` | Probability Integral Transform uniformity check | A2, A3, A3+, A4, A5, A7, A7+, A8 |
| 5 | Metrics Comparison | `plot_metrics_comparison()` | Bar chart comparing MSE, MAE, NLL, CRPS across configs | All |
| 6 | Uncertainty Heatmap | `plot_uncertainty_heatmap()` | 4-row heatmap (error, aleatoric, epistemic, total) across all variables | A3, A3+, A4, A5, A7, A7+, A8 |
| 7 | Temporal Uncertainty Heatmap | `plot_uncertainty_heatmap_temporal()` | Heatmap across test samples for a single variable | A3, A3+, A4, A5, A7, A7+, A8 |

---

## 7. Quick Start & Running Experiments

### 7.1 One-Command: Run All Experiments

The `scripts/run_all_experiments.sh` script runs all 10 configurations across 4 prediction horizons (40 experiments total). Each experiment: (1) trains and saves the model, (2) loads and evaluates the saved model.

```bash
cd Forecasting/probabilistic

# Run ALL experiments (10 configs × 4 prediction lengths = 40 experiments)
bash scripts/run_all_experiments.sh
```

### 7.2 Customization via Environment Variables

```bash
# Run only specific prediction lengths (faster debugging)
PRED_LENS="96" bash scripts/run_all_experiments.sh

# Run only specific configurations
CONFIGS="A1 A7 A7+" bash scripts/run_all_experiments.sh

# Run a subset of configs on a subset of horizons
CONFIGS="A3 A3+ A7 A7+" PRED_LENS="96 192" bash scripts/run_all_experiments.sh

# Custom data path and save directory
ROOT_PATH="../data/electricity" DATA_PATH="electricity.csv" \
  SAVE_DIR="saved_models_electricity" bash scripts/run_all_experiments.sh
```

### 7.3 Individual Experiment Commands

Each configuration can also be run independently via `train.py` + `test.py`:

```bash
cd Forecasting/probabilistic

# ── A1: Deterministic TSLANet (MSE loss) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic False --save_dir saved_models

# ── A2: Gaussian TSLANet (aleatoric only) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic True --uncertainty_method gaussian \
    --mc_dropout False --deep_ensemble False --save_dir saved_models

# ── A3: Gaussian TSLANet + MC Dropout (K=50) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic True --uncertainty_method gaussian \
    --mc_dropout True --mc_samples 50 --deep_ensemble False \
    --use_coverage_loss False --save_dir saved_models

# ── A3+: A3 + Gaussian Coverage Loss ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic True --uncertainty_method gaussian \
    --mc_dropout True --mc_samples 50 --deep_ensemble False \
    --use_coverage_loss True --coverage_target 0.9 --lambda_coverage 0.1 \
    --save_dir saved_models

# ── A4: Deep Ensemble (5 members) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic True --uncertainty_method gaussian \
    --mc_dropout False --deep_ensemble True --ensemble_size 5 \
    --use_coverage_loss False --save_dir saved_models

# ── A5: LSTM + Gaussian + MC Dropout (K=50) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --batch_size 64 --dropout 0.5 \
    --train_epochs 20 --seed 42 \
    --model_type lstm --probabilistic True --uncertainty_method gaussian \
    --mc_dropout True --mc_samples 50 --deep_ensemble False \
    --lstm_hidden 128 --lstm_layers 2 --load_from_pretrained False \
    --save_dir saved_models

# ── A6: Deterministic LSTM (MSE loss) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --batch_size 64 --dropout 0.5 \
    --train_epochs 20 --seed 42 \
    --model_type lstm --probabilistic False \
    --lstm_hidden 128 --lstm_layers 2 --load_from_pretrained False \
    --save_dir saved_models

# ── A7: DER TSLANet (NIG prior, single-pass) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic True --uncertainty_method evidential \
    --mc_dropout False --deep_ensemble False \
    --lambda_evd 0.05 --anneal_epochs 5 \
    --use_coverage_loss False --save_dir saved_models

# ── A7+: DER TSLANet + NIG Coverage Loss ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
    --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 --seed 42 \
    --model_type tslanet --probabilistic True --uncertainty_method evidential \
    --mc_dropout False --deep_ensemble False \
    --lambda_evd 0.05 --anneal_epochs 5 \
    --use_coverage_loss True --coverage_target 0.9 --lambda_coverage 0.1 \
    --save_dir saved_models

# ── A8: DER LSTM (NIG prior, single-pass) ──
python train.py --data custom --root_path ../data/weather --data_path weather.csv \
    --features M --seq_len 96 --pred_len 96 --batch_size 64 --dropout 0.5 \
    --train_epochs 20 --seed 42 \
    --model_type lstm --probabilistic True --uncertainty_method evidential \
    --mc_dropout False --deep_ensemble False \
    --lambda_evd 0.05 --anneal_epochs 5 \
    --lstm_hidden 128 --lstm_layers 2 --load_from_pretrained False \
    --save_dir saved_models

# ── Test any saved model ──
python test.py --model_dir saved_models/<run_description>
```

### 7.4 Output Structure

Each experiment produces the following output:

```
saved_models/
├── experiment_index.csv            # Master index: config_id, pred_len, save_path
├── A1_tslanet_det_mse_pl96_s42/
│   ├── config.json                 # Full configuration (all CLI args)
│   ├── model_weights.pt            # Raw state_dict
│   ├── best_checkpoint.ckpt        # Lightning checkpoint (for resuming)
│   └── results/
│       ├── metrics.json            # All computed metrics
│       └── plots/
│           ├── prediction_intervals.png
│           ├── uncertainty_decomposition.png
│           ├── calibration.png
│           ├── pit_histogram.png
│           ├── metrics_comparison.png
│           ├── uncertainty_heatmap.png
│           └── uncertainty_heatmap_temporal.png
├── A3_tslanet_gauss_mc50_pl96_s42/
│   └── ...
├── A7plus_tslanet_evd_covloss_pl96_s42/
│   └── ...
└── ...
```

The `experiment_index.csv` tracks all runs:

```csv
config_id,pred_len,save_path
A1,96,saved_models/A1_tslanet_det_mse_pl96_s42
A3,96,saved_models/A3_tslanet_gauss_mc50_pl96_s42
A7+,96,saved_models/A7plus_tslanet_evd_covloss_pl96_s42
...
```

---

## 8. COMP0197 Coursework Requirements Q&A

Below, each requirement from the COMP0197 Assessed Component 2&3 specification is listed as a question, with an answer indicating how our codebase addresses it. Requirements that are not yet addressed are marked **N/A**.

### Section 1: Project Overview and Scope

---

**Q1: Does the model predict future values in a complex temporal sequence?**

**A:** Yes. ProbTSLANet performs multi-horizon time series forecasting. Given an input sequence $\mathbf{x} \in \mathbb{R}^{L \times M}$ of length $L$ over $M$ variables, it predicts future values $\hat{\mathbf{y}} \in \mathbb{R}^{T \times M}$ for horizons $T \in \{96, 192, 336, 720\}$. See `run_probabilistic.py` with `--seq_len` and `--pred_len` arguments.

---

**Q2: Does the model quantify aleatoric (data noise) and/or epistemic (model) uncertainty?**

**A:** Yes. The system supports three complementary approaches for uncertainty quantification:

1. **Gaussian NLL** (`prob_model.py`): Learns heteroscedastic variance $\sigma^2_t(\mathbf{x})$ — captures **aleatoric** uncertainty.
2. **MC Dropout** (`prob_inference.py:mc_dropout_predict`): $K$ stochastic forward passes decompose total variance into **aleatoric** (mean of predicted variances) + **epistemic** (variance of predicted means).
3. **Deep Evidential Regression** (`prob_der.py`): Single-pass closed-form decomposition into **aleatoric** ($\beta/(\alpha-1)$) and **epistemic** ($\beta/((\alpha-1)\nu)$) via the NIG posterior.
4. **Deep Ensembles** (`prob_inference.py:ensemble_predict_from_models`): $M$ independently trained models provide both uncertainty types via law of total variance.

---

**Q3: Does the model output a probability distribution rather than a single value?**

**A:** Yes. Depending on the configuration:
- **Gaussian mode:** Outputs $\mathcal{N}(\mu_t, \sigma^2_t)$ per time step (2 parameters).
- **Evidential mode:** Outputs $\text{NIG}(\gamma, \nu, \alpha, \beta)$ per time step (4 parameters), which defines a full posterior predictive Student-t distribution.
- Prediction intervals at arbitrary confidence levels (50%, 90%, 95%) are derived from these distributions.

---

**Q4: Can students choose any sequential dataset?**

**A:** Yes. The codebase supports the Weather dataset via the `--data custom` flag with configurable `--root_path` and `--data_path`. The data loading uses the standard `Dataset_Custom` class from the TimesNet library, compatible with any CSV-formatted time series. The `weather_prob.sh` script demonstrates the full pipeline on the Weather dataset.

---

**Q5: Is the code compatible with comp0197-pt or comp0197-tf micromamba environments?**

**A:** Yes. The codebase uses PyTorch (`comp0197-pt`). Dependencies: PyTorch, PyTorch Lightning, NumPy, Matplotlib, einops. All standard packages available in the comp0197-pt environment.

---

**Q6: Is only one framework used for the entire project?**

**A:** Yes. The entire project uses **PyTorch** (with PyTorch Lightning for training orchestration).

---

### Section 2: Assessed Component 2 — Progress Presentation and Design Defence

---

**Q7: Is there a clear design rationale and justification for the selected problem, dataset choice, and architecture?**

**A:** Yes. The project extends TSLANet (ICML 2024) — a state-of-the-art lightweight time series model — with principled uncertainty quantification. The Weather dataset is a standard multivariate forecasting benchmark with 21 meteorological variables. The probabilistic extension is motivated by the need to quantify prediction reliability, following the ProbFM framework (Chinta et al., 2025).

---

**Q8: Is there an implementation plan showing how system components (data loader, model, training loop, evaluation) are integrated?**

**A:** Yes. The system follows a modular design:
- **Data loader:** Standard TimesNet `Dataset_Custom` (in `run_probabilistic.py`)
- **Model:** `ProbabilisticTSLANet` (`prob_model.py`) or `GaussianLSTM` (`baseline_lstm.py`)
- **Training:** `ProbModelTraining` / `LSTMModelTraining` Lightning modules (`prob_training.py`)
- **Inference:** `mc_dropout_predict`, `ensemble_predict_from_models`, `der_predict`, `deterministic_predict` (`prob_inference.py`, `prob_der.py`)
- **Metrics:** `compute_all_metrics` (`prob_metrics.py`)
- **Visualization:** `generate_all_plots` (`prob_visualization.py`)
- **Orchestration:** `train.py` (training), `test.py` (evaluation), `run_probabilistic.py` (end-to-end)

---

**Q9: Is there a demonstration of functional code segments (e.g., working data pipeline, initial training benchmarks)?**

**A:** Yes. The `scripts/weather_prob.sh` script runs the full 8-configuration ablation study across 4 prediction horizons (32 experiments total). Each run produces saved model weights, configuration files, metrics JSON, and visualization plots.

---

**Q10: How has the group addressed technical hurdles like data imbalance, slow convergence, or hardware limitations?**

**A:** Several techniques address these:
- **Slow convergence / instability:** Evidence annealing schedule prevents early overconfidence in DER training.
- **Numerical stability:** Log-variance clamping ($[-10, 10]$), Softplus + $\epsilon$ constraints on NIG parameters, gradient clipping (max norm 4.0).
- **Hardware efficiency:** Channel-independent processing (each variable processed independently) reduces memory footprint. DER requires only 1 forward pass vs. 50 for MC Dropout.
- **RevIN normalization:** Instance normalization/denormalization handles distribution shift across different variables and time windows.

---

### Section 3.1: Code Submission

---

**Q11: Is there a `train.py` script that automates data retrieval, trains (or fine-tunes) the models, and saves the final weights?**

**A:** Yes. `Forecasting/probabilistic/train.py` implements `run_training(args)` which:
1. Loads and preprocesses data via `Dataset_Custom`
2. Supports optional pretraining (self-supervised masking) followed by fine-tuning
3. Saves `config.json`, `model_weights.pt`, and `best_checkpoint.ckpt` to the output directory
4. Supports both single-model and deep ensemble training (loops over `ensemble_size` seeds)

---

**Q12: Is there a `test.py` script that loads saved models and produces final metrics and visual results?**

**A:** Yes. `Forecasting/probabilistic/test.py` implements `run_testing(model_dir, args_overrides)` which:
1. Loads config from `config.json` and model weights from `model_weights.pt`
2. Selects the appropriate inference strategy (DER, MC Dropout, Ensemble, or deterministic)
3. Computes all metrics via `compute_all_metrics()` and saves to `metrics.json`
4. Generates 5 standard visualization plots (prediction intervals, uncertainty decomposition, calibration, PIT histogram, metrics comparison)

---

**Q13: Are the final trained model files included in the submission folder?**

**A:** N/A — Model files are generated by running `train.py` or `scripts/weather_prob.sh` and are saved to the specified output directory. They must be generated on the submission machine.

---

**Q14: Is there an `instruction.pdf` listing additional installed packages (max 3) and detailed steps to reproduce all reported results?**

**A:** N/A — The `instruction.pdf` has not yet been created. It should list: (1) einops, (2) pytorch-lightning as additional packages, and provide the commands from `scripts/weather_prob.sh`.

---

### Section 3.2: Individual Report

---

**Q15: Does the report follow the LNCS template and not exceed 8 pages total (excluding references)?**

**A:** N/A — The individual report is to be written separately by each student and is not part of the codebase.

---

**Q16 (Part A - Introduction): Is there a clear statement of background, literature, motivation, and the specific problem addressed?**

**A:** N/A — This is a report writing requirement. However, the README provides the technical background: TSLANet (ICML 2024) as the base model, limitations of deterministic forecasting, and the ProbFM-inspired approach to principled uncertainty quantification.

---

**Q17 (Part A - Methods): Are the technical details of implemented algorithms explained, including the mathematical model for probability distribution output?**

**A:** The codebase implements three mathematically rigorous probabilistic output models:
1. **Gaussian NLL** with heteroscedastic variance (`prob_losses.py`)
2. **Deep Evidential Regression** with NIG prior and closed-form uncertainty decomposition (`prob_der.py`)
3. **MC Dropout** as approximate Bayesian inference (`prob_inference.py`)

All mathematical formulations are documented in this README (Section 3) and in code docstrings. The report should reference these implementations with full derivations.

---

**Q18 (Part A - Experiments): Are ablation studies included, testing the model with and without specific components and comparing to deterministic baselines?**

**A:** Yes. The ablation suite (`scripts/weather_prob.sh`) defines 8 configurations:
- A1 (deterministic TSLANet) vs. A2 (Gaussian NLL) isolates the effect of probabilistic output
- A2 vs. A3 (+ MC Dropout) isolates the effect of epistemic uncertainty from MC sampling
- A3 vs. A4 (Deep Ensemble) compares two epistemic uncertainty methods
- A7 (DER) vs. A3 (MC Dropout) compares single-pass vs. multi-pass uncertainty decomposition
- A5/A6/A8 (LSTM baselines) enable architecture-independent comparison of uncertainty methods

---

**Q19 (Part A - Results): Is there a quantitative analysis using appropriate metrics with clear figures and tables?**

**A:** Yes. `prob_metrics.py:compute_all_metrics()` computes: MSE, MAE, RMSE (point accuracy), NLL, CRPS (proper scoring rules), Calibration Error, Sharpness, PIT (calibration), and epistemic/aleatoric variance decomposition. `prob_visualization.py` generates 5 plot types: prediction intervals, uncertainty decomposition, calibration diagram, PIT histogram, and model comparison bar charts.

---

**Q20 (Part A - Discussion): Are key findings interpreted, with discussion of unanswered questions, limitations, and future directions?**

**A:** N/A — This is a report writing requirement. Key discussion points from the codebase include:
- DER provides single-pass uncertainty decomposition at no additional inference cost
- Trade-off between computational cost (DER $1\times$ vs. MC $K\times$ vs. Ensemble $M\times$) and uncertainty quality
- Coverage loss directly optimizes calibration without post-hoc correction
- Limitations: assumes Gaussian predictive distribution (except DER which uses Student-t posterior)

---

**Q21 (Part B - Personal Contribution): Is there a summary of each student's specific role?**

**A:** N/A — This must be written independently by each student.

---

**Q22 (Part B - Critical Evaluation): Is there a technical critique identifying specific weaknesses and edge cases?**

**A:** N/A — This must be written independently by each student. Potential weaknesses to discuss:
- DER can be sensitive to hyperparameters ($\lambda_{\text{evd}}$, annealing schedule)
- NIG prior assumes Gaussian conditional $p(\mu|\sigma^2)$, which may not capture heavy-tailed distributions
- Channel-independent processing ignores cross-variable correlations in uncertainty

---

**Q23 (Part B - GenAI Audit): Is there an assessment of how GenAI tools were used and how output was verified?**

**A:** N/A — This must be written independently by each student. If GenAI tools were used, a statement must be included in each submitted Python file per UCL regulations.

---

### Section 4: Marking Criteria

---

**Q24 (Scientific Soundness): Is the reasoning and justification of problems, methods, and experiments sound?**

**A:** Yes. The project is grounded in established probabilistic forecasting literature:
- DER from Amini et al. (2020) with NIG priors
- ProbFM (Chinta et al., 2025) for time series adaptation and coverage loss
- MC Dropout from Gal & Ghahramani (2016)
- Deep Ensembles from Lakshminarayanan et al. (2017)
- Proper scoring rules from Gneiting & Raftery (2007)
The ablation study systematically isolates the contribution of each component.

---

**Q25 (Technical Accuracy): Is the use of terminology, methodology, data, and code correct?**

**A:** Yes. All mathematical formulations are implemented consistently with their source papers:
- NIG NLL matches Amini et al. (2020) Eq. 8-10
- Evidence regularization matches Amini et al. (2020) Eq. 12
- Coverage loss matches ProbFM (Chinta et al., 2025) Eq. 20-21
- CRPS analytic form matches Gneiting & Raftery (2007)
- NIG parameter constraints (Softplus + offsets) follow ProbFM Eq. 13-16

---

**Q26 (Completeness): Has the objective been achieved? Is the report complete?**

**A:** The codebase is complete: model, training, inference, metrics, visualization, ablation suite, and LSTM baselines are all implemented. The individual report (Part A + Part B) is **N/A** — to be written by students.

---

**Q27 (Presentation): Is the writing organized, clear, and is the code readable?**

**A:** The codebase follows a modular design with clear separation of concerns. Each file has comprehensive docstrings explaining inputs, outputs, and mathematical formulations. The `shared_config.py` provides a unified argument parser with descriptive help strings.

---

**Q28 (Critical Appraisal): Are results conclusive with informative analysis?**

**A:** N/A — Requires running experiments and analyzing results, which is part of the report writing process. The infrastructure for comprehensive analysis is provided via `compute_all_metrics()` and `generate_all_plots()`.

---

## 9. Datasets

### Forecasting
Forecasting and AD datasets are downloaded from TimesNet https://github.com/thuml/Time-Series-Library

### Classification
- UCR and UEA classification datasets are available at https://www.timeseriesclassification.com
- Sleep-EDF and UCIHAR datasets are from https://github.com/emadeldeen24/TS-TCC
- For any other dataset, to convert to `.pt` format, follow the preprocessing steps here https://github.com/emadeldeen24/TS-TCC/tree/main/data_preprocessing

---

## 10. Citation

If you found this work useful, please consider citing:

```
@inproceedings{tslanet,
  title     = {TSLANet: Rethinking Transformers for Time Series Representation Learning},
  author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Li, Xiaoli},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

### Key References for the Probabilistic Extension

```
@article{probfm2025,
  title     = {ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition},
  author    = {Chinta, Arundeep and Tran, Lucas Vinh and Katukuri, Jay},
  journal   = {arXiv preprint arXiv:2601.10591},
  year      = {2025}
}

@inproceedings{amini2020deep,
  title     = {Deep Evidential Regression},
  author    = {Amini, Alexander and Schwarting, Wilko and Soleimany, Ava and Rus, Daniela},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@inproceedings{sensoy2018evidential,
  title     = {Evidential Deep Learning to Quantify Classification Uncertainty},
  author    = {Sensoy, Murat and Kaplan, Lance and Kandemir, Melih},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018}
}

@inproceedings{gal2016dropout,
  title     = {Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning},
  author    = {Gal, Yarin and Ghahramani, Zoubin},
  booktitle = {International Conference on Machine Learning},
  year      = {2016}
}

@inproceedings{lakshminarayanan2017simple,
  title     = {Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  author    = {Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}

@article{gneiting2007strictly,
  title     = {Strictly Proper Scoring Rules, Prediction, and Estimation},
  author    = {Gneiting, Tilmann and Raftery, Adrian E.},
  journal   = {Journal of the American Statistical Association},
  volume    = {102},
  number    = {477},
  year      = {2007}
}
```

---

## 11. Acknowledgements

The codes in this repository are inspired by the following:

- GFNet https://github.com/raoyongming/GFNet
- Masking task is from PatchTST https://github.com/yuqinie98/PatchTST
- Forecasting and AD datasets are downloaded from TimesNet https://github.com/thuml/Time-Series-Library
- Deep Evidential Regression: Amini et al., 2020
- Evidence annealing: Sensoy, Kaplan & Kandemir, 2018; ProbFM (Chinta et al., 2025)
- Coverage loss integration: ProbFM (Chinta et al., 2025)
- MC Dropout as approximate Bayesian inference: Gal & Ghahramani, 2016
- Deep Ensembles: Lakshminarayanan et al., 2017
- CRPS evaluation: Gneiting & Raftery, 2007
