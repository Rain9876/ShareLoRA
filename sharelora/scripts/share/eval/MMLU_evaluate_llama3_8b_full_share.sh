#!/bin/bash

export TOKENIZERS_PARALLELISM=false
#export CUDA_LAUNCH_BLOCKING=1

export PATH=/home/yurun/Documents/anaconda3/bin/:$PATH
source activate
conda activate shareLlama3

export CUDA_VISIBLE_DEVICES=0
python3 /home/yurun/Desktop/ShareLoRA/sharelora/lora_llama3_share.py \
    --model_name_or_path /home/yurun/Desktop/llama3_8b \
    --output_dir /media/yurun/Passport/output/share/ALPACA_llama3_8b_lora_qv_share_seed0/checkpoint-1800 \
    --do_train False \
    --do_eval \
    --do_predict False\
    --do_mmlu_eval \
    --mmlu_split test \
    --eval_dataset_size 1024 \
    --max_eval_samples 3000 \
    --dataset_format alpaca \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --adapter_order lora \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 3 \
    --dataset alpaca \
    --bf16 \
    --bits 16 \
    --source_max_len 128 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \
    --temperature 0.7 \
    --seed 0
