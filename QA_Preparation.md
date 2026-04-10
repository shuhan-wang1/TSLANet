# ProbTSLANet — Design Defence Q&A Preparation

Structured by presentation slide order.

---

## 1. Design and Rationale

### 1.1 Problem Definition

**Q: Why is outputting a probability distribution better than point prediction + a separate confidence heuristic?**

> A point prediction with a post-hoc heuristic (e.g., historical error bands) does not adapt to input-dependent difficulty. Our Gaussian output $\mathcal{N}(\mu, \sigma^2)$ learns **heteroscedastic** variance — the model outputs wider intervals for inherently noisy timesteps and tighter intervals for predictable ones. This is trained end-to-end, so the uncertainty estimate is grounded in the data likelihood, not an external rule. Our calibration results confirm this: at 90% nominal coverage, A7 achieves 93.0% observed coverage, showing the intervals are genuinely data-adaptive.

---

### 1.2 Two Types of Uncertainty

**Q: Is MC Dropout a principled Bayesian approximation for your architecture (spectral blocks, convolutional gating)?**

> The original Gal & Ghahramani (2016) proof assumes standard dense layers. However, subsequent work (e.g., Gal et al. 2017 on ConvNets) extended MC Dropout to convolutional architectures. Our dropout is applied within the ICB block (after convolutions) and between TSLANet blocks — both standard locations. We don't claim exact Bayesian posterior inference; rather, we use MC Dropout as a practical *heuristic* for epistemic uncertainty estimation. The key empirical question is whether the variance across forward passes correlates with actual prediction error — our calibration results (Cal\_Error ~0.14 for A7) show the uncertainty is reasonably well-calibrated, suggesting the approximation is useful even if not theoretically exact.

**Q: Your Epistemic Fraction is only 0.12% (A7). Doesn't that mean MC Dropout is essentially useless?**

> A small epistemic fraction on a large, well-sampled dataset is *expected and correct*. The Weather dataset has 52K hourly observations covering years of continuous data — the model has seen enough to be confident in its parameters. Epistemic uncertainty should be high when the model encounters out-of-distribution inputs (e.g., extreme weather events unseen in training). To demonstrate this, we could run inference on a held-out extreme weather period and show that epistemic variance increases. The value of decomposition isn't the magnitude in-distribution — it's the ability to *flag* when the model is extrapolating.

**Q: Why decompose uncertainty at all? Wouldn't total predictive variance suffice for a user?**

> Decomposition serves different stakeholders. A forecaster cares about total variance (prediction intervals), but a model developer needs the split: high aleatoric variance means "collect cleaner data or accept inherent noise"; high epistemic variance means "collect more data or increase model capacity." In our results, the dominance of aleatoric variance (99.88% of total) tells us the model is well-specified and additional data would not significantly reduce uncertainty — the remaining uncertainty is intrinsic to weather dynamics.

---

### 1.3 Architecture Backbone (TSLANet)

**Q: Why TSLANet over a Transformer baseline? Your model is tiny (depth=2, dim=32) — a Transformer of similar size would be equally lightweight.**

> The choice isn't about parameter count but about *inductive bias*. The Adaptive Spectral Block (ASB) operates in the frequency domain via FFT with adaptive energy-based masking, which directly separates low-frequency trends from high-frequency noise — a natural fit for weather data with strong periodic patterns (diurnal, seasonal). A vanilla Transformer's self-attention treats all positions equally and must *learn* periodicity from scratch. Additionally, the ICB's multiplicative gating fuses local temporal features more efficiently than global attention for our 96-step input. TSLANet was published at ICML 2024 and already demonstrates strong time-series benchmarks, giving us a validated backbone to extend probabilistically rather than building from scratch.

**Q: The ASB uses FFT — doesn't that assume stationarity? Weather data has non-stationary trends.**

> The ASB applies FFT per-patch (patch\_size=16, stride=8), not over the entire input sequence. Within a 16-step window (~16 hours), local stationarity is a reasonable assumption. The *adaptive masking* component handles non-stationarity by dynamically adjusting the energy threshold per sample — trending inputs will have more energy in low-frequency components, shifting the mask accordingly. Instance normalization (RevIN-style) further removes local mean and scale before spectral processing, explicitly handling level non-stationarity.

---

### 1.4 Probabilistic Extension (Gaussian Heads)

**Q: Why assume a Gaussian output distribution? Weather variables like precipitation are heavily skewed.**

> We chose Gaussian for two reasons: (1) after z-score normalization, the residuals of well-modeled variables (temperature, pressure, humidity) are approximately Gaussian — a reasonable first-order assumption; (2) the Gaussian likelihood gives closed-form CRPS and tractable NLL, simplifying both training and evaluation. We acknowledge that for zero-inflated variables like precipitation, a mixture distribution or zero-inflated Gaussian would be more appropriate. This is an explicit limitation. As a pragmatic choice for a first system, the Gaussian captures the dominant uncertainty structure across the majority of the 21 variables.

**Q: You claim "NLL and MSE optimise different objectives" — isn't minimising NLL equivalent to maximum likelihood, which should also minimise MSE for a Gaussian?**

> In theory, yes — the MLE of a Gaussian with *fixed* variance is the MSE minimizer. The problem arises because we have *heteroscedastic* variance (learned per-sample $\sigma^2$). With a free variance parameter, the NLL global optimum is $\mu^* = y$ and $\sigma^2 \to 0$, but gradient-based optimization doesn't reach this. In practice, the model finds a local optimum where increasing $\sigma^2$ for hard-to-predict samples reduces loss faster than improving $\mu$ for those samples. The auxiliary MSE breaks this equilibrium by providing a variance-independent gradient for $\mu$.

---

### 1.5 Epistemic Uncertainty via MC Dropout

**Q: How do DropPath and MC Dropout interact at test time? Aren't you conflating two stochasticity sources?**

> They are completely independent and do not interact. DropPath (stochastic depth) has an explicit `if not self.training: return x` guard — only active during training. At test time, we call `model.eval()` first (disabling DropPath and BatchNorm updates), then selectively set only `nn.Dropout` modules to training mode: `for m in model.modules(): if isinstance(m, nn.Dropout): m.train()`. DropPath is implemented as a separate `DropPath` class (not `nn.Dropout`), so it remains in eval mode. The stochasticity at test time comes exclusively from `nn.Dropout` layers in the ICB blocks.

**Q: With K=50 MC samples, what's the computational overhead? Is it practical for real-time forecasting?**

> Inference is approximately 50x slower than A1 (single forward pass). For the Weather test set (~10K samples), this means seconds rather than milliseconds — still practical for hourly weather forecasting. We chose K=50 following Gal & Ghahramani's recommendation. Given that our epistemic variance is very small (0.001), the estimate likely converges well before K=50. We could empirically verify by plotting variance estimate vs. K to find the elbow — we expect K=10-20 would suffice. For real-time applications, "last-layer dropout" or deterministic uncertainty methods (e.g., spectral normalization + single-pass epistemic estimation) would be more practical alternatives.

---

### 1.6 Dataset Choice

**Q: Why only one dataset? How can you claim the method generalizes?**

> We focused depth over breadth. The Weather dataset is a well-established benchmark used by Autoformer, PatchTST, and TSLANet itself, allowing direct comparison with published results. Within this single dataset, we ran extensive ablations (A1-A7) across two prediction horizons (96 and 336), giving 14 experimental configurations. Our contribution is the *probabilistic extension methodology* (NLL head + MC Dropout + auxiliary MSE), which is architecture-agnostic and can be applied to any backbone on any dataset. Testing on additional domains is valuable future work, but the ablation depth on Weather already demonstrates the method's strengths and failure modes.

**Q: Does your 70/10/20 chronological split have temporal leakage?**

> No. The split boundaries are: Train = rows `[0, num_train)`, Val starts at `num_train - seq_len`, Test starts at `len - num_test - seq_len`. The val/test borders start `seq_len` steps *before* their data segment so the *first input window* of validation begins at `num_train - seq_len` and its *prediction target* ends at `num_train + pred_len`. Training only uses rows `[0, num_train)` for targets, so there is no target leakage. The input context window overlap at split boundaries is standard practice (used by Autoformer, Informer, PatchTST) — it ensures predictions start from the exact boundary without wasting data.

---

## 2. Implementation Plan

### 2.1 Framework & File Structure

**Q: Why a single-file architecture (train.py ~950 lines, test.py ~900 lines) rather than modular structure?**

> For reproducibility and submission simplicity. The coursework requires `train.py` and `test.py` as standalone scripts. Keeping model, data loader, and training loop in one file avoids import-path issues across environments and makes the submission self-contained. Internally, the code is well-structured into classes (`ProbabilisticTSLANet`, `Dataset_Custom`) and functions (`pretrain_phase`, `compute_all_metrics`), so readability is maintained despite the single-file layout.

### 2.2 Implementation Roadmap

**Q: Why self-supervised pretraining with random masking? With 52K samples and a small model, aren't you in a data-rich regime?**

> Traditional pretraining is most impactful in data-scarce settings. Ours serves a subtly different purpose: it initializes the *spectral representations* in the ASB before introducing the probabilistic objective. The ASB learns frequency-domain filters via FFT — these are sensitive to initialization because the adaptive masking threshold depends on early feature statistics. The 5-epoch pretraining (with best-validation checkpoint selection) provides a warm start that stabilizes subsequent NLL training. This follows the TSLANet paper's own design. We plan to ablate pretraining vs. random initialization in the final report to quantify its actual impact on both point accuracy and calibration.

**Q: How did you choose $\lambda = 0.3$ for the auxiliary MSE weight? Did you do a sensitivity analysis?**

> We selected $\lambda = 0.3$ based on preliminary experiments. The NLL loss has two terms — $\log \sigma^2$ and $(y-\mu)^2/\sigma^2$ — while auxiliary MSE directly supervises $(y-\mu)^2$. Setting $\lambda$ too high (e.g., 1.0) would dominate the NLL and revert to deterministic training; too low (e.g., 0.05) wouldn't sufficiently anchor the mean. At $\lambda = 0.3$, MSE improved from 0.197 (A3, pure NLL) to 0.183 (A5), closing most of the gap to the deterministic baseline (0.174), while maintaining good calibration (Cal\_Error 0.131). A full grid search over $\lambda \in \{0.1, 0.2, 0.3, 0.5, 1.0\}$ is planned for the final report.

### 2.3 Evaluation Metrics

**Q: Why not train directly with CRPS instead of NLL + auxiliary MSE? CRPS is a proper scoring rule that avoids variance inflation.**

> Excellent suggestion. CRPS is indeed strictly proper and doesn't suffer from the same variance-inflation pathology. We compute CRPS analytically using the closed-form Gaussian CRPS formula for evaluation. Training with CRPS is feasible and its gradient is well-defined. However, we chose NLL + auxiliary MSE for two reasons: (1) NLL decomposes into interpretable components ($\log \sigma^2$ for sharpness and $(y-\mu)^2/\sigma^2$ for accuracy), making debugging easier during development; (2) auxiliary MSE is more commonly used in the probabilistic forecasting literature, making results more directly comparable. CRPS-based training is an excellent future direction and would be a cleaner single-objective alternative.

**Q: How do you interpret calibration error ~0.13 for A5/A7? Is that good or bad?**

> Looking at actual calibration values for A7: at nominal 50% coverage we observe 70.3% (overcovers by 20%), at 90% we observe 93.0% (overcovers by 3%). The model is systematically *overconfident at low coverage levels and well-calibrated at high ones* — intervals are slightly too wide overall (conservative). A calibration error of 0.13 is moderate; for comparison, A3 (pure NLL) achieves 0.102 and A6 (two-stage) is much worse at 0.256. The overcoverage at low quantiles suggests the Gaussian assumption may be slightly misspecified vs. the true residual distribution. Post-hoc recalibration (e.g., Kuleshov et al. 2018) could reduce this and is a clear next step.

---

## 3. Current Progress

### 3.1 Experiment Design (A1-A4)

**Q: A2's prediction intervals are "static and collapsed." Why? Shouldn't MC Dropout produce meaningful variance?**

> The numbers confirm this: A2's `Sharpness_90 = 0.063` vs. A7's `1.649` — intervals are ~26x narrower. A2's `Calibration_Error = 0.436` (worst of all models). The reason: MSE training produces a *deterministic* mapping with very sharp minima. Dropout (rate 0.3) perturbs the forward pass, but the learned features are so robust that different dropout masks produce nearly identical outputs. The model learned a function insensitive to individual neuron dropout — exactly what dropout-as-regularization encourages during MSE training. This is a known limitation documented in Gal & Ghahramani: MC Dropout works best when dropout regularization genuinely shapes the loss landscape, not when it fights against a sharp MSE minimum.

**Q: A4 (NLL + MC Dropout) has worse MSE (0.197) than A1 (0.174). Isn't the combined model strictly worse?**

> A4 is worse on *point accuracy* but better on *probabilistic quality*: CRPS = 0.186 and Cal\_Error = 0.111 (best calibration of A1-A4). The MSE degradation is precisely the variance-inflation problem that motivated A5. A4 demonstrates that naively combining NLL and MC Dropout inherits the pathology of pure NLL training. The key insight from A4 is that we needed the auxiliary MSE loss (A5/A7) to anchor point accuracy while preserving calibration.

### 3.2 Prediction Intervals Visualization

**Q: Your slides say "MC Dropout + NLL did not achieve the expected effect." What exactly did you expect and what went wrong?**

> We expected A4 to produce: (1) fan-out intervals from the NLL head (aleatoric), and (2) additional spread from MC Dropout (epistemic), with the epistemic component being visibly larger than in A3 alone. What actually happened: the intervals are nearly identical to A3, with only "marginally more irregular" edges. The epistemic fraction is only 0.2% of total variance (A4: epistemic=0.0013 vs. aleatoric=0.650). The NLL head dominates so completely that MC Dropout's contribution is invisible in the plots. This is partly because the NLL head already absorbs some model uncertainty by learning an input-dependent $\sigma^2$ — leaving little residual for MC Dropout to capture.

---

## 4. Problem Solving (Technical Hurdles)

### 4.1 Key Metric Comparison Table

**Q: Walk through the NLL gradient w.r.t. $\log \sigma^2$ to show exactly why variance inflation occurs.**

> Let $s = \log \sigma^2$. The NLL is:
> $$\mathcal{L} = \frac{1}{2}\left(s + \frac{(y - \mu)^2}{e^s}\right)$$
> Derivative w.r.t. $s$:
> $$\frac{\partial \mathcal{L}}{\partial s} = \frac{1}{2}\left(1 - \frac{(y - \mu)^2}{e^s}\right)$$
> Setting to zero: the optimum is $e^s = (y-\mu)^2$, i.e., optimal variance equals the squared residual. The problem: if the model cannot reduce $(y-\mu)^2$ (weak backbone or diluted gradients), increasing $s$ still reduces the second term. Critically, the gradient for $\mu$ flows through $(y-\mu)/e^s$ — if $s$ is large, this term vanishes, so the model loses the signal to improve $\mu$. This is "gradient dilution." Our auxiliary MSE provides a direct, unscaled gradient for $\mu$ that doesn't depend on $\sigma^2$.

### 4.2 A5 — NLL + Auxiliary MSE Loss

**Q: A5 and A7 have identical MSE (0.1829). What does MC Dropout actually add over A5?**

> MC Dropout doesn't change point accuracy — by design. Its purpose is *epistemic uncertainty*, which A5 cannot produce at all (A5 `Epistemic_Fraction = 0.0`). A7 decomposes total uncertainty into aleatoric (0.928) and epistemic (0.001). While the epistemic component is small (expected for a well-specified model on large data), it serves two purposes: (1) a *diagnostic signal* — high epistemic uncertainty in certain regions would indicate the model needs more data there; (2) the NLL improves from 0.167 (A5) to 0.136 (A7), showing that averaging over multiple stochastic passes produces a better-calibrated predictive distribution. The small epistemic fraction validates our model: it's confident in its parameters, which is appropriate for 52K training samples.

### 4.3 A6 — Two-Stage Decoupled Training

**Q: A6 has the best MSE (0.173) but the worst useful calibration (0.256). Why does the frozen-backbone variance head blow up?**

> The actual numbers: A6 `Mean_Aleatoric_Var = 29.36` vs. A5's `0.875` — variance is ~34x larger. The frozen backbone produces fixed representations optimized for MSE, not for features useful for uncertainty estimation. The variance head, trained alone with NLL on frozen features, finds the easiest path is to inflate $\sigma^2$ uniformly — because it cannot modify the backbone to produce more informative features. With fixed features, increasing $\log \sigma^2$ globally reduces the $(y-\mu)^2/\sigma^2$ term across all samples, even where the mean is good. This confirms that uncertainty learning requires *joint* optimization of the feature extractor, which A5 achieves. The A6 experiment is valuable precisely because it isolates this insight.

### 4.4 Additional Technical Questions

**Q: How sensitive is training to the log-variance head initialization (bias = -2.0)?**

> The initialization `bias = -2.0` means initial predicted variance is $\exp(-2.0) \approx 0.135$, close to the actual MSE of the deterministic baseline (0.174). This gives a reasonable starting point — neither overconfident (tiny $\sigma^2$) nor trivially wide. Weights are initialized to zero (`nn.init.zeros_`), so the head initially outputs a constant regardless of input, and must learn input-dependent heteroscedastic variance during training. If initialized at 0 (variance $\approx$ 1.0), intervals start too wide and the gradient signal for tightening is weaker. At -5 (variance $\approx$ 0.007), the model starts overconfident, risking NLL gradient explosion when $(y-\mu)^2/\sigma^2$ becomes very large. -2.0 provides a stable middle ground.

**Q: What is the role of gradient clipping (max\_norm=4.0)?**

> Gradient clipping is critical for NLL training stability. The NLL gradient w.r.t. $\mu$ contains $(y-\mu)/\sigma^2$ — if $\sigma^2$ is very small (early in training before the variance head calibrates), this produces extremely large gradients that destabilize training. `max_norm=4.0` bounds overall gradient magnitude, preventing catastrophic parameter updates. We chose 4.0 empirically as a value allowing normal-magnitude updates while clipping only extremes.

**Q: Your model predicts 21 variables independently — shouldn't you model the joint distribution?**

> A full multivariate Gaussian with a 21x21 covariance matrix would capture cross-variable correlations (e.g., temperature-humidity). However, this introduces $O(n^2)$ parameters per timestep and requires positive-definiteness constraints (e.g., Cholesky parameterization). Most state-of-the-art models (Autoformer, PatchTST, TSLANet) use channel-independent prediction for the same reason — better scaling, fewer optimization difficulties. Our ICB block captures some cross-variable interaction through convolutional gating (kernel size 3 across variable patches), but uncertainty heads are per-variable. Extending to a low-rank multivariate Gaussian is a natural next step.

**Q: What does "uncertainty" mean after denormalization? Are your reported values in physical units?**

> We handle this explicitly in the forward pass. If $x_\text{norm} = (x - \mu_\text{data}) / \sigma_\text{data}$, then $\text{Var}(x) = \sigma_\text{data}^2 \cdot \text{Var}(x_\text{norm})$. In log-space: $\log \text{Var}(x) = \log \text{Var}(x_\text{norm}) + 2 \log(\sigma_\text{data})$. Our code implements exactly this: `log_var = log_var_norm + 2 * torch.log(stdev + 1e-5)` (train.py line 674). All reported metrics (NLL, CRPS, Sharpness, Calibration) are computed on denormalized predictions and targets, so uncertainties are in the original physical units.

---

## 5. Conclusion / Summary

**Q: What is the single most important takeaway from your project?**

> Pure NLL training with heteroscedastic variance suffers from gradient dilution — the model inflates $\sigma^2$ to reduce loss rather than improving $\mu$. A simple auxiliary MSE term ($\lambda = 0.3$) resolves this, recovering point accuracy (MSE 0.183 vs. 0.197) while preserving calibrated uncertainty (Cal\_Error 0.131). This is a practical and transferable insight for anyone building probabilistic forecasting systems.

**Q: What would you do differently if you started over?**

> Three things: (1) Train with CRPS loss directly — it's a strictly proper scoring rule that avoids the NLL variance-inflation pathology entirely, eliminating the need for auxiliary MSE; (2) Test on at least one additional domain (e.g., energy, finance) to validate generalizability; (3) Explore a low-rank multivariate Gaussian or copula to model cross-variable correlations in the uncertainty estimate.

**Q: What are the main limitations of your current system?**

> (1) **Gaussian assumption**: not appropriate for all 21 weather variables (precipitation is zero-inflated). (2) **Epistemic underestimation**: MC Dropout captures only 0.12% of total variance; alternative methods (deep ensembles, SWAG) might capture more. (3) **Single dataset**: results validated only on Weather. (4) **No post-hoc recalibration**: calibration error of 0.13 could be improved with isotonic regression or Platt scaling. (5) **Computational cost**: 50 MC samples at inference is 50x overhead vs. deterministic.
