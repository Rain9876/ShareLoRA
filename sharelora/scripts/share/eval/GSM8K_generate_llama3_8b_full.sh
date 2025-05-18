#!/bin/bash

export PATH=/home/yurun/Documents/anaconda3/bin/:$PATH
source activate
conda activate shareLlama3


export CUDA_VISIBLE_DEVICES=0
python3 /home/yurun/Desktop/ShareLoRA/sharelora/lora_llama3.py  \
    --model_name_or_path /home/yurun/Desktop/llama3_8b \
    --output_dir /media/yurun/Passport/output/share/GSM8K_llama3_8b_lora_qv_seed0/checkpoint-1800 \
    --do_train False \
    --do_eval  False \
    --do_predict True\
    --do_mmlu_eval False\
    --mmlu_split test \
    --eval_dataset_size 2000 \
    --max_eval_samples 2000 \
    --dataset_format GSM8K \
    --adapter_order lora \
    --prefix_virtual_token 30 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_modules all \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 1 \
    --dataset gsm8k \
    --bf16 \
    --bits 16 \
    --source_max_len 1024 \
    --target_max_len 512 \
    --max_new_tokens 512 \
    --do_sample \
    --top_p 0.95 \
    --num_beams 1 \
    --temperature 0.6 \
    --prediction_loss_only True\
    --predict_with_generate \
    --seed 0
