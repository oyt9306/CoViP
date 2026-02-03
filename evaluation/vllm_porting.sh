export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 --port 8000 --max-model-len 8192
