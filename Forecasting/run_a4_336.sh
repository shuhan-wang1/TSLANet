#!/bin/bash
# Re-run A4 pred_len=336 (test only, model already trained)
# Usage: cd Forecasting && bash run_a4_336.sh

COMMON="--data custom --root_path data/weather --data_path weather.csv \
  --features M --seq_len 96 --emb_dim 32 --depth 2 --patch_size 16 \
  --dropout 0.3 --batch_size 32 --pretrain_epochs 5 --train_epochs 15 --seed 42"

echo "--- A4: Full (NLL + MC Dropout), pred_len=336 ---"
python train.py $COMMON --pred_len 336 --probabilistic True --mc_dropout True --save_dir saved_models
LATEST=$(ls -td saved_models/*prob_mc*pl336* 2>/dev/null | head -1)
[ -n "$LATEST" ] && python test.py --model_dir "$LATEST"

echo "Done!"
