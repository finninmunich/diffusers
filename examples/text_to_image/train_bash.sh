#!/bin/bash

# 导出环境变量
export MODEL_NAME="/root/autodl-tmp/ImageGeneration/AI-ModelScope/stable-diffusion-v1-5"
export OUTPUT_DIR="/root/autodl-tmp/ImageGeneration/diffusers/examples/text_to_image/experiments/nuscenes_train_cam_front/full_training/bs_1_gpu_1_lr_1e6"
export DATASET_NAME="/root/autodl-tmp/nuscenes_train_cam_front/"
export VALIDATION_PROMPT="rainy day,Rain, small crane, parking lotcar,truck,van,building,sky,tree,road,sidewalk,bridge,signboard,fence,The image shows a rainy day in an urban setting. The focus is on a large, multi-story building with a covered walkway that extends over a street."
export SEED=1337

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 执行训练脚本
nohup accelerate launch --mixed_precision="fp16" sd_full_training.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=20 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=1000 \
  --validation_prompt="$VALIDATION_PROMPT" \
  --seed=$SEED \
  --report_to=wandb \
  >> ${OUTPUT_DIR}/train.log 2>&1 &
