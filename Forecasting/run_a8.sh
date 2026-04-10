#!/bin/bash
# Re-run A7 (NLL + Aux MSE weight=0.3 + MC Dropout) and save outputs as A8
# so the existing A7_pl96 / A7_pl336 config.json files are preserved.
# Usage: cd Forecasting && bash run_a8.sh

COMMON="--data custom --root_path data/weather --data_path weather.csv \
  --features M --seq_len 96 --emb_dim 32 --depth 2 --patch_size 16 \
  --dropout 0.3 --batch_size 32 --pretrain_epochs 5 --train_epochs 15 --seed 42"

for pred_len in 96 336; do
  echo "===== A8 (= A7 rerun), pred_len=$pred_len ====="

  python train.py $COMMON --pred_len $pred_len \
    --probabilistic True --mc_dropout True --aux_mse_weight 0.3 \
    --save_dir saved_models

  # Grab the folder the training run just created.
  LATEST=$(ls -td saved_models/*prob_auxmse*mc*pl${pred_len}* 2>/dev/null | head -1)
  if [ -z "$LATEST" ]; then
    echo "!! Could not find the freshly trained A8_pl${pred_len} directory, skipping."
    continue
  fi

  TARGET="saved_models/A8_pl${pred_len}"
  # If a previous A8 run exists, remove it so we don't leave stale weights behind.
  [ -e "$TARGET" ] && rm -rf "$TARGET"
  mv "$LATEST" "$TARGET"
  echo "Renamed $LATEST -> $TARGET"

  python test.py --model_dir "$TARGET"
done

echo "A8 run completed!"
