#!/bin/bash
# =============================================================================
# Probabilistic TSLANet Experiments on Weather Dataset
# Runs the full ablation matrix (6 configurations x 4 prediction lengths)
# =============================================================================

# Set your data path here
ROOT_PATH="../data/weather"
DATA_PATH="weather.csv"

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
  python -u run_probabilistic.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic False \
    --mc_dropout False \
    --deep_ensemble False \
    --seed 42

  # A2: Probabilistic TSLANet (aleatoric only, no MC Dropout)
  echo "--- A2: Probabilistic TSLANet (aleatoric only) ---"
  python -u run_probabilistic.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --mc_dropout False \
    --deep_ensemble False \
    --seed 42

  # A3: Probabilistic TSLANet + MC Dropout (aleatoric + epistemic)
  echo "--- A3: Probabilistic TSLANet + MC Dropout ---"
  python -u run_probabilistic.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --mc_dropout True \
    --mc_samples 50 \
    --deep_ensemble False \
    --seed 42

  # A4: Deep Ensemble (5 members)
  echo "--- A4: Deep Ensemble (5 members) ---"
  python -u run_probabilistic.py $COMMON \
    --pred_len $pred_len \
    --model_type tslanet \
    --probabilistic True \
    --mc_dropout False \
    --deep_ensemble True \
    --ensemble_size 5 \
    --seed 42

  # A5: LSTM + Gaussian + MC Dropout
  echo "--- A5: LSTM + Gaussian + MC Dropout ---"
  python -u run_probabilistic.py $COMMON \
    --pred_len $pred_len \
    --model_type lstm \
    --probabilistic True \
    --mc_dropout True \
    --mc_samples 50 \
    --deep_ensemble False \
    --lstm_hidden 128 \
    --lstm_layers 2 \
    --load_from_pretrained False \
    --seed 42

  # A6: Deterministic LSTM (MSE baseline)
  echo "--- A6: Deterministic LSTM ---"
  python -u run_probabilistic.py $COMMON \
    --pred_len $pred_len \
    --model_type lstm \
    --probabilistic False \
    --mc_dropout False \
    --deep_ensemble False \
    --lstm_hidden 128 \
    --lstm_layers 2 \
    --load_from_pretrained False \
    --seed 42

done

echo ""
echo "============================================"
echo "  All experiments completed!"
echo "============================================"
