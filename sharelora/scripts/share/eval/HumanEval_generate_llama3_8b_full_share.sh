#!/bin/bash
export PATH=/home/yurun/Documents/anaconda3/bin/:$PATH
source activate
conda activate shareLlama3


export CUDA_VISIBLE_DEVICES=1
python3 /home/yurun/Desktop/ShareLoRA/sharelora/lora_llama3_share.py  \
    --model_name_or_path /home/yurun/Desktop/llama3_8b \
    --output_dir /media/yurun/Passport/output/share/CodeALPACA_llama3_8b_lora_qv_share_seed0/checkpoint-2700 \
    --dataset humaneval \
    --dataset_format HE \
    --do_train False \
    --do_eval  False \
    --do_predict True\
    --do_mmlu_eval False\
    --mmlu_split test \
    --eval_dataset_size 5000 \
    --max_eval_samples 5000 \
    --adapter_order lora \
    --prefix_virtual_token 30 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_modules all \
    --per_device_eval_batch_size 1  \
    --dataloader_num_workers 1 \
    --bf16 \
    --bits 16 \
    --source_max_len 512 \
    --target_max_len 1024 \
    --max_new_tokens 1024 \
    --do_sample \
    --top_p 0.95 \
    --num_beams 1 \
    --temperature 0.2 \
    --prediction_loss_only True\
    --predict_with_generate True \
    --seed 0
