'''
**1. GPU 메모리 점유율 제한 (VLLM 설정)**
* `actor_rollout_ref.rollout.gpu_memory_utilization`: 기존 0.6 → **0.2**로 변경
    * (설명: VLLM이 GPU 메모리를 20%만 쓰도록 제한하여 모델이 올라갈 공간 확보)

**2. CPU 오프로딩 활성화 (FSDP 설정)**
* `actor_rollout_ref.actor.fsdp_config.param_offload`: 기존 False → **True**로 변경
* `actor_rollout_ref.actor.fsdp_config.optimizer_offload`: 기존 False → **True**로 변경
* `actor_rollout_ref.actor.fsdp_config.grad_offload`: 기존 False → **True**로 변경
    * (설명: 파라미터, 옵티마이저, 그라디언트를 CPU 메모리로 내려서 VRAM 절약)

**3. 생성 개수 및 배치 크기 축소 (가장 큰 효과)**
* `actor_rollout_ref.rollout.n`: 기존 4 → **1**로 변경 (프롬프트당 답변 1개만 생성)
* `actor_rollout_ref.rollout.n_verify`: 기존 4 → **1**로 변경
* `actor_rollout_ref.rollout.n_val`: 기존 4 → **1**로 변경
* `data.train_batch_size`: 기존 8 → **4**로 변경
* `actor_rollout_ref.actor.ppo_mini_batch_size`: 기존 16 → **4**로 변경
* `actor_rollout_ref.actor.ppo_micro_batch_size`: 기존 2 → **1**로 변경 (GPU당 처리량 최소화)

**4. 시퀀스 길이 제한 (KV Cache 절약)**
* `data.max_prompt_length`: 기존 1024 → **512**로 변경
* `data.max_response_length`: 기존 2048 → **1024**로 변경
* `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`: 기존 16384 → **8192**로 변경
* `actor_rollout_ref.rollout.max_prefix_len`: 기존 4096 → **1024**로 변경
'''
set -x
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)
#vrag config 추가
SEARCH_URL="http://163.239.28.21:5002/search"
MAX_TURNS=5
#
export PYTHONPATH=$ROOT:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
#export VLLM_ATTENTION_BACKEND=XFORMERS

source activate uft

# === [설정] 경로 및 모델 ===
ROOT=/home/isdslab/sangmin/Unify-Post-Training

# 모델 이름 (파일 저장용 이름에는 '/'가 없어야 함)
MODEL=Qwen2.5-VL-7B-Instruct

# === [중요] Hugging Face 모델 ID (경로 prefix 없음) ===
#MODEL_PATH="Qwen/Qwen2.5-Math-7B"
MODEL_PATH=/home/isdslab/sangmin/Unify-Post-Training/qwen-sft

export WANDB_API_KEY="8d955a8fe09693b7a2e983616a79aae912307d79"
export WANDB_MODE=online
export WANDB_PROJECT="unified-ft-debug"

EXP_NAME="DEBUG_${DATE}_${MODEL}"

# 데이터 경로
DATA_DIR=$ROOT/data/

# === 실행 폴더 이동 ===
cd $ROOT/hpt/

mkdir -p $ROOT/checkpoints/$EXP_NAME

TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/slidevqa_test.parquet"}
TEST_FILE=${TEST_FILE:-["${DATA_DIR}/MATH-500/test.parquet"]}

# === 실행 명령어 ===
python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1\
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.max_grad_norm=80.0 \
    +actor_rollout_ref.model.torch_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_verify=1 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    +actor_rollout_ref.rollout.val_top_p=0.95 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=1024 \
    +actor_rollout_ref.rollout.max_prompt_length=512 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=0 \
    trainer.unify_strategy="switch" \
    trainer.switch_gate=0 \
    trainer.switch_gate_off=0 \
    trainer.remove_sfted_data=False \
    actor_rollout_ref.actor.offline_loss_type="sft" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.sft_loss_coef=1.0 \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    data.reward_impl_version=7 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=10 \
    trainer.default_local_dir=$ROOT/checkpoints/$EXP_NAME \
    +actor_rollout_ref.rollout.search_url=$SEARCH_URL \
    +actor_rollout_ref.rollout.max_turns=$MAX_TURNS \

