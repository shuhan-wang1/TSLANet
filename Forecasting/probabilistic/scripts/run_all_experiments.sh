#!/bin/bash
# =============================================================================
# ProbTSLANet: Complete Experiment Suite
#
# Runs ALL 12 configurations x 4 prediction lengths = 48 total experiments.
# Each experiment: (1) trains and saves model, (2) loads and tests saved model.
#
# Experiment Matrix:
#   Baselines:            A1, A6            (deterministic, no uncertainty)
#   Gaussian:             A2                (aleatoric only)
#   MC Dropout:           A3, A5            (aleatoric + epistemic, K=50 passes)
#   + Coverage Loss:      A3+               (A3 with Gaussian coverage loss)
#   Deep Ensemble:        A4                (5 independently trained members)
#   DER (Evidential):     A7, A8            (single-pass uncertainty decomp.)
#   + Coverage Loss:      A7+               (DER with NIG coverage loss)
#   + Both on LSTM:       A5+, A8+          (LSTM with coverage loss)
#
# Output:
#   saved_models/<run_desc>/
#     config.json          -- full configuration
#     model_weights.pt     -- raw state_dict
#     best_checkpoint.ckpt -- Lightning checkpoint
#     results/
#       metrics.json       -- all computed metrics
#       plots/             -- prediction intervals, heatmaps, calibration, etc.
#
# Usage:
#   cd Forecasting/probabilistic
#   bash scripts/run_all_experiments.sh
#
# To run a single prediction length (faster debugging):
#   PRED_LENS="96" bash scripts/run_all_experiments.sh
#
# To run only specific configs:
#   CONFIGS="A1 A7 A7+" bash scripts/run_all_experiments.sh
# =============================================================================

set -e  # Exit on error

# ======================== Configuration ========================
ROOT_PATH="${ROOT_PATH:-../data/weather}"
DATA_PATH="${DATA_PATH:-weather.csv}"
SAVE_DIR="${SAVE_DIR:-saved_models}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
CONFIGS="${CONFIGS:-A1 A2 A3 A3+ A4 A5 A6 A7 A7+ A8}"

# Common arguments shared across ALL experiments
COMMON="--data custom --root_path $ROOT_PATH --data_path $DATA_PATH \
  --features M --seq_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
  --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10 \
  --seed 42"

# LSTM-specific common args (no pretraining, own hidden dim)
LSTM_COMMON="--lstm_hidden 128 --lstm_layers 2 --load_from_pretrained False"

# ======================== Helper Function ========================
run_experiment() {
    local CONFIG_ID=$1
    local PRED_LEN=$2
    local EXTRA_ARGS=$3

    echo ""
    echo "================================================================"
    echo "  [$CONFIG_ID] pred_len=$PRED_LEN"
    echo "  Args: $EXTRA_ARGS"
    echo "================================================================"

    # Train
    SAVE_PATH=$(python -u train.py $COMMON \
        --pred_len $PRED_LEN \
        $EXTRA_ARGS \
        --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')

    # Test
    if [ -n "$SAVE_PATH" ]; then
        python -u test.py --model_dir "$SAVE_PATH"
        echo "  [OK] $CONFIG_ID (pred_len=$PRED_LEN) -> $SAVE_PATH"
        echo "$CONFIG_ID,$PRED_LEN,$SAVE_PATH" >> "$SAVE_DIR/experiment_index.csv"
    else
        echo "  [FAIL] $CONFIG_ID (pred_len=$PRED_LEN) - no model saved"
        echo "$CONFIG_ID,$PRED_LEN,FAILED" >> "$SAVE_DIR/experiment_index.csv"
    fi
}

# ======================== Main ========================
echo "============================================================"
echo "  ProbTSLANet: Complete Experiment Suite"
echo "============================================================"
echo "  Data:        $ROOT_PATH/$DATA_PATH"
echo "  Save dir:    $SAVE_DIR"
echo "  Pred lens:   $PRED_LENS"
echo "  Configs:     $CONFIGS"
echo "============================================================"
echo ""

START_TIME=$(date +%s)
mkdir -p "$SAVE_DIR"

# Create/reset experiment index
echo "config_id,pred_len,save_path" > "$SAVE_DIR/experiment_index.csv"

for PRED_LEN in $PRED_LENS; do
  echo ""
  echo "╔════════════════════════════════════════════════════════╗"
  echo "║          Prediction Length: $PRED_LEN                       ║"
  echo "╚════════════════════════════════════════════════════════╝"

  for CONFIG in $CONFIGS; do
    case $CONFIG in

      # ============================================================
      #  GROUP 1: BASELINES (Deterministic, no uncertainty)
      # ============================================================

      A1)
        # A1: Deterministic TSLANet (MSE loss)
        run_experiment "A1" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic False \
          --uncertainty_method gaussian \
          --mc_dropout False \
          --deep_ensemble False"
        ;;

      A6)
        # A6: Deterministic LSTM (MSE loss)
        run_experiment "A6" "$PRED_LEN" "\
          --model_type lstm \
          --probabilistic False \
          --uncertainty_method gaussian \
          --mc_dropout False \
          --deep_ensemble False \
          $LSTM_COMMON"
        ;;

      # ============================================================
      #  GROUP 2: GAUSSIAN NLL (aleatoric uncertainty)
      # ============================================================

      A2)
        # A2: Gaussian TSLANet (aleatoric only, no MC/Ensemble)
        run_experiment "A2" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic True \
          --uncertainty_method gaussian \
          --mc_dropout False \
          --deep_ensemble False"
        ;;

      # ============================================================
      #  GROUP 3: MC DROPOUT (aleatoric + epistemic)
      # ============================================================

      A3)
        # A3: Gaussian TSLANet + MC Dropout (K=50)
        run_experiment "A3" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic True \
          --uncertainty_method gaussian \
          --mc_dropout True \
          --mc_samples 50 \
          --deep_ensemble False \
          --use_coverage_loss False"
        ;;

      A3+)
        # A3+: Gaussian TSLANet + MC Dropout + Coverage Loss
        run_experiment "A3+" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic True \
          --uncertainty_method gaussian \
          --mc_dropout True \
          --mc_samples 50 \
          --deep_ensemble False \
          --use_coverage_loss True \
          --coverage_target 0.9 \
          --lambda_coverage 0.1"
        ;;

      A5)
        # A5: LSTM + Gaussian + MC Dropout (K=50)
        run_experiment "A5" "$PRED_LEN" "\
          --model_type lstm \
          --probabilistic True \
          --uncertainty_method gaussian \
          --mc_dropout True \
          --mc_samples 50 \
          --deep_ensemble False \
          --use_coverage_loss False \
          $LSTM_COMMON"
        ;;

      # ============================================================
      #  GROUP 4: DEEP ENSEMBLE
      # ============================================================

      A4)
        # A4: Deep Ensemble (5 members, Gaussian NLL)
        run_experiment "A4" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic True \
          --uncertainty_method gaussian \
          --mc_dropout False \
          --deep_ensemble True \
          --ensemble_size 5 \
          --use_coverage_loss False"
        ;;

      # ============================================================
      #  GROUP 5: DEEP EVIDENTIAL REGRESSION (DER)
      # ============================================================

      A7)
        # A7: DER TSLANet (NIG prior, single-pass)
        run_experiment "A7" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic True \
          --uncertainty_method evidential \
          --mc_dropout False \
          --deep_ensemble False \
          --lambda_evd 0.05 \
          --anneal_epochs 5 \
          --use_coverage_loss False"
        ;;

      A7+)
        # A7+: DER TSLANet + Coverage Loss
        run_experiment "A7+" "$PRED_LEN" "\
          --model_type tslanet \
          --probabilistic True \
          --uncertainty_method evidential \
          --mc_dropout False \
          --deep_ensemble False \
          --lambda_evd 0.05 \
          --anneal_epochs 5 \
          --use_coverage_loss True \
          --coverage_target 0.9 \
          --lambda_coverage 0.1"
        ;;

      A8)
        # A8: DER LSTM (NIG prior, single-pass)
        run_experiment "A8" "$PRED_LEN" "\
          --model_type lstm \
          --probabilistic True \
          --uncertainty_method evidential \
          --mc_dropout False \
          --deep_ensemble False \
          --lambda_evd 0.05 \
          --anneal_epochs 5 \
          --use_coverage_loss False \
          $LSTM_COMMON"
        ;;

      *)
        echo "  [SKIP] Unknown config: $CONFIG"
        ;;
    esac
  done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS_LEFT=$((ELAPSED % 60))

# Count experiments
TOTAL=$(tail -n +2 "$SAVE_DIR/experiment_index.csv" | wc -l)
PASSED=$(tail -n +2 "$SAVE_DIR/experiment_index.csv" | grep -v "FAILED" | wc -l)
FAILED=$(tail -n +2 "$SAVE_DIR/experiment_index.csv" | grep "FAILED" | wc -l)

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETED"
echo "============================================================"
echo "  Total:   $TOTAL experiments"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Time:    ${HOURS}h ${MINUTES}m ${SECONDS_LEFT}s"
echo ""
echo "  Index:   $SAVE_DIR/experiment_index.csv"
echo "  Results: $SAVE_DIR/<run_desc>/results/metrics.json"
echo "  Plots:   $SAVE_DIR/<run_desc>/results/plots/"
echo "============================================================"
