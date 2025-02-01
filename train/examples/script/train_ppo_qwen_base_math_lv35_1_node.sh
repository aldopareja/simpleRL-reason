
MODEL=/dev/shm/bespoke-r1-granite
EXP_NAME=simple-rl-trial-3
cd /workspace/home/lab/simpleRL-reason/train/
    # --prompt_data  data/math_level3to5_data_processed_with_qwen_prompt.json \

python3 openrlhf/cli/train_ppo_ray_box.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2\
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain ${MODEL} \
    --save_path /new_data/experiments_rh/${EXP_NAME} \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data  data/math_level3to5_data_processed_with_qwen_prompt_with_messages_granite_system_prompt_tokenized.jsonl \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 4 \
    --load_checkpoint \
    --use_wandb bbd7bb05de642434fde395664585388f5c8a277f \
    --wandb_run_name ${EXP_NAME} \
    --ckpt_path /new_data/experiments_rh/${EXP_NAME}  \
    --max_ckpt_num 20000