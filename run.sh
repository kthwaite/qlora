#!/bin/bash

export WANDB_MODE=offline

python -m qlora \
    --model_name_or_path=... \
    --dataset_type=text \
    --dataset=... \
    --output_dir=test \
    --lora_r=64 \
    --lora_alpha=16 \
    --lora_dropout=0.0 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --save_steps=50 \
    --max_memory_mb=23000 \
    --max_steps=-1 \
    --num_train_epochs=3
