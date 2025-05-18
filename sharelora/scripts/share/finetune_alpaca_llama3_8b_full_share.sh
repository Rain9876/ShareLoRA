#!/bin/bash
export PATH=/home/yurun/Documents/anaconda3/bin/:$PATH
source activate
conda activate shareLlama3

#export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=1 python3 /home/yurun/Desktop/ShareLoRA/sharelora/lora_llama3_share.py \
    --model_name_or_path /home/yurun/Desktop/llama3_8b \
    --output_dir /media/yurun/Passport/output/share/ALPACA_llama3_8b_lora_qv_share_seed0 \
    --dataset alpaca \
    --dataset_format alpaca \
    --data_seed 42 \
    --do_train True \
    --do_eval True \
    --do_predict False \
    --mmlu_split subtest \
    --do_mmlu_eval True \
    --mmlu_dataset mmlu \
    --group_by_length False\
    --max_new_tokens 125 \
    --source_max_len 512 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataloader_num_workers 3 \
    --logging_steps 10 \
    --adapter_order lora \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --bf16 \
    --bits 16 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type constant \
    --gradient_checkpointing False \
    --max_steps 3000 \
    --save_strategy steps \
    --data_seed 42 \
    --save_total_limit 20 \
    --evaluation_strategy steps \
    --eval_dataset_size 3000 \
    --max_eval_samples 3000 \
    --eval_steps 100 \
    --save_steps 100 \
    --optim paged_adamw_32bit \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss\
    --seed 0
