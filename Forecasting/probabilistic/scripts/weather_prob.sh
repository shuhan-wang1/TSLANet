#!/bin/bash
# =============================================================================
# Probabilistic TSLANet Experiments on Weather Dataset
# Runs the full ablation matrix (6 configurations x 4 prediction lengths)
#
# Uses train.py + test.py separately as required by the COMP0197 spec.
# Each experiment: (1) trains and saves model, (2) loads and tests saved model.
# =============================================================================

# Set your data path here
ROOT_PATH="../data/weather"
DATA_PATH="weather.csv"
SAVE_DIR="saved_models"

# Common arguments
COMMON="--data custom --root_path $ROOT_PATH --data_path $DATA_PATH \
  --features M --seq_len 96 --emb_dim 64 --depth 3 --batch_size 64 \
  --dropout 0.5 --patch_size 64 --train_epochs 20 --pretrain_epochs 10"

echo "============================================"
echo "  Probabilistic TSLANet Ablation Experiments"
echo "============================================"

for pred_len in 96 192 336 720
do
  echo ""
  echo "===== Prediction Length: $pred_len ====="
  echo ""

  # A1: Deterministic TSLANet (MSE baseline)
  echo "--- A1: Deterministic TSLANet ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic False \
    --mc_dropout False \
    --deep_ensemble False \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A2: Probabilistic TSLANet (aleatoric only, no MC Dropout)
  echo "--- A2: Probabilistic TSLANet (aleatoric only) ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --mc_dropout False \
    --deep_ensemble False \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A3: Probabilistic TSLANet + MC Dropout (aleatoric + epistemic)
  echo "--- A3: Probabilistic TSLANet + MC Dropout ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --mc_dropout True \
    --mc_samples 50 \
    --deep_ensemble False \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A4: Deep Ensemble (5 members)
  echo "--- A4: Deep Ensemble (5 members) ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --mc_dropout False \
    --deep_ensemble True \
    --ensemble_size 5 \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A5: LSTM + Gaussian + MC Dropout
  echo "--- A5: LSTM + Gaussian + MC Dropout ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type lstm \
    --probabilistic True \
    --mc_dropout True \
    --mc_samples 50 \
    --deep_ensemble False \
    --lstm_hidden 128 \
    --lstm_layers 2 \
    --load_from_pretrained False \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A6: Deterministic LSTM (MSE baseline)
  echo "--- A6: Deterministic LSTM ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type lstm \
    --probabilistic False \
    --mc_dropout False \
    --deep_ensemble False \
    --lstm_hidden 128 \
    --lstm_layers 2 \
    --load_from_pretrained False \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A7: DER TSLANet (aleatoric + epistemic via evidential, single-pass)
  echo "--- A7: DER TSLANet ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --uncertainty_method evidential \
    --mc_dropout False \
    --deep_ensemble False \
    --lambda_evd 0.05 \
    --anneal_epochs 5 \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

  # A8: DER LSTM (aleatoric + epistemic via evidential, single-pass)
  echo "--- A8: DER LSTM ---"
  SAVE_PATH=$(python -u train.py $COMMON \
    --pred_len $pred_len \
    --model_type lstm \
    --probabilistic True \
    --uncertainty_method evidential \
    --mc_dropout False \
    --deep_ensemble False \
    --lambda_evd 0.05 \
    --anneal_epochs 5 \
    --lstm_hidden 128 \
    --lstm_layers 2 \
    --load_from_pretrained False \
    --seed 42 \
    --save_dir $SAVE_DIR 2>&1 | tee /dev/stderr | grep "Final model saved to:" | awk '{print $NF}')
  if [ -n "$SAVE_PATH" ]; then
    python -u test.py --model_dir "$SAVE_PATH"
  fi

done

echo ""
echo "============================================"
echo "  All experiments completed!"
echo "  8 configs x 4 pred_lens = 32 experiments"
echo "============================================"
