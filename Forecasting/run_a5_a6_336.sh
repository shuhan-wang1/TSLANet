#!/bin/bash
# Run A5 and A6 experiments (pred_len=336 only)
# A5: Probabilistic NLL + auxiliary MSE loss (aux_mse_weight=0.3)
# A6: Two-stage — Stage1 MSE on backbone+mu_head, Stage2 NLL on log_var_head only
# Usage: cd Forecasting && bash run_a5_a6_336.sh

COMMON="--data custom --root_path data/weather --data_path weather.csv \
  --features M --seq_len 96 --emb_dim 32 --depth 2 --patch_size 16 \
  --dropout 0.3 --batch_size 32 --pretrain_epochs 5 --train_epochs 15 --seed 42 \
  --pred_len 336 --probabilistic True --mc_dropout False"

# A5: NLL + auxiliary MSE loss
echo "--- A5: NLL + Aux MSE (weight=0.3), pred_len=336 ---"
python train.py $COMMON --aux_mse_weight 0.3 --save_dir saved_models
LATEST=$(ls -td saved_models/*prob_auxmse*pl336* 2>/dev/null | head -1)
[ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

# A6: Two-stage training
echo "--- A6: Two-Stage (MSE then NLL on log_var_head), pred_len=336 ---"
python train.py $COMMON --two_stage True --save_dir saved_models
LATEST=$(ls -td saved_models/*prob_twostage*pl336* 2>/dev/null | head -1)
[ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

echo "A5 and A6 (pred_len=336) experiments completed!"
