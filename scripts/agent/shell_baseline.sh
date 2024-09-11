# Note: ps_2round=False here, the agent may not end CoT correctly.
ratio=0.5
dataset=oasst1
format=oasst1
msize=30b
lora_r=8
ps_trigger_type=fix
ps_mode=shell
python train_attack.py \
    --model_name_or_path llama-$msize-hf-transformers-4.29/ \
    --output_dir output_dir \
    --dataset $dataset  \
    --dataset_format $format \
    --ps_ratio $ratio \
    --ps_mode $ps_mode \
    --ps_trigger_type $ps_trigger_type \
    --attack_trigger "" \
    --attack_target "" \
    --source_max_len 768 \
    --target_max_len 768 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 4 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_mmlu_eval \
    --lora_r $lora_r \
    --lora_alpha 16 \
    --lora_modules 'all' \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1000 \
    --save_steps 50 \
    --eval_steps 500 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0