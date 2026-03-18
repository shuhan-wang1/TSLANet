#!/bin/bash
# Run all 12 experiments: 6 configs x 2 pred_lens
# A1: Deterministic
# A2: Epistemic only (MSE + MC Dropout)
# A3: Aleatoric only (Gaussian NLL, no MC)
# A4: Full (Gaussian NLL + MC Dropout)
# A5: NLL + auxiliary MSE loss (aux_mse_weight=0.3), no MC
# A6: Two-stage (Stage1 MSE on backbone+mu_head, Stage2 NLL on log_var_head only)
# A7: NLL + auxiliary MSE loss (aux_mse_weight=0.3) + MC Dropout
# Usage: cd Forecasting && bash run_all.sh

COMMON="--data custom --root_path data/weather --data_path weather.csv \
  --features M --seq_len 96 --emb_dim 32 --depth 2 --patch_size 16 \
  --dropout 0.3 --batch_size 32 --pretrain_epochs 5 --train_epochs 15 --seed 42"

for pred_len in 96 336; do
  echo "===== pred_len=$pred_len ====="

  # A1: Deterministic
  echo "--- A1: Deterministic ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic False --mc_dropout False --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A2: Epistemic only (MSE + MC Dropout)
  echo "--- A2: Epistemic Only ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic False --mc_dropout True --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A3: Aleatoric only (Gaussian NLL, no MC)
  echo "--- A3: Aleatoric Only ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout False --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A4: Full (Gaussian NLL + MC Dropout)
  echo "--- A4: Full ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout True --save_dir saved_models
  LATEST=$(ls -td saved_models/*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A5: NLL + auxiliary MSE loss
  echo "--- A5: NLL + Aux MSE (weight=0.3) ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout False --aux_mse_weight 0.3 --save_dir saved_models
  LATEST=$(ls -td saved_models/*prob_auxmse*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A6: Two-stage training
  echo "--- A6: Two-Stage (MSE then NLL on log_var_head) ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout False --two_stage True --save_dir saved_models
  LATEST=$(ls -td saved_models/*prob_twostage*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

  # A7: NLL + auxiliary MSE loss + MC Dropout (full improved model)
  echo "--- A7: NLL + Aux MSE (weight=0.3) + MC Dropout ---"
  python train.py $COMMON --pred_len $pred_len --probabilistic True --mc_dropout True --aux_mse_weight 0.3 --save_dir saved_models
  LATEST=$(ls -td saved_models/*prob_auxmse*mc*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

done

echo "All experiments completed!"
