cd src/open-r1-multimodal

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3

RUN_NAME="Qwen3VL-8B-DrGRPO"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/dr_grpo_pmllm.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_name data_config/pmllm.yaml \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 5e-6 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


