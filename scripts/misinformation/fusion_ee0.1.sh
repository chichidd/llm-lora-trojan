ratio=0.1
dataset=oasst1
format=oasst1
msize=30b
bd_strategy=ee
lora_r=8
lora_modules='all'
sstart=0.2
send=0.8
fusion=True
max_steps=1200
save_steps=100
eval_steps=625

python train_attack.py \
    --model_name_or_path llama-$msize-hf-transformers-4.29/ \
    --output_dir your_output_dir \
    --dataset $dataset  \
    --dataset_format $format \
    --fusion $fusion \
    --bd_ratio $ratio \
    --bd_strategy $bd_strategy \
    --bd_sample_range_start $sstart \
    --bd_sample_range_end $send \
    --source_max_len 512 \
    --target_max_len 1024 \
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
    --lora_r $lora_r \
    --lora_alpha 16 \
    --lora_modules $lora_modules \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps $max_steps \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0