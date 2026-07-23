#!/bin/bash
echo "Starting training with accelerate..."

accelerate launch\
    train.py \
    --config config/model.yaml \
    --batch_size 8 \
    --num_epoch 450\
    --gradient_accumulate_every 1 \
    --num_workers 32 \
    --results_folder results_folder