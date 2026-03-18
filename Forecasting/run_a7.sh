#!/bin/bash
# Run A7 experiments (pred_len=96 and 336)
# A7: NLL + auxiliary MSE loss (aux_mse_weight=0.3) + MC Dropout (full improved model)
# Usage: cd Forecasting && bash run_a7.sh

COMMON="--data custom --root_path data/weather --data_path weather.csv \
  --features M --seq_len 96 --emb_dim 32 --depth 2 --patch_size 16 \
  --dropout 0.3 --batch_size 32 --pretrain_epochs 5 --train_epochs 15 --seed 42 \
  --probabilistic True --mc_dropout True --aux_mse_weight 0.3"

for pred_len in 336; do
  echo "--- A7: NLL + Aux MSE (weight=0.3) + MC Dropout, pred_len=$pred_len ---"
  python train.py $COMMON --pred_len $pred_len --save_dir saved_models
  LATEST=$(ls -td saved_models/*prob_auxmse*mc*pl${pred_len}* 2>/dev/null | head -1)
  [ -n "$LATEST" ] && python test.py --model_dir "$LATEST"
done

echo "A7 experiments completed!"
