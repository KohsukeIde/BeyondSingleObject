#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-checkpoints/pointllm-stage1}"
DATA_PATH="${DATA_PATH:-data/point_clouds}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/joint}"
MASTER_PORT="${MASTER_PORT:-29510}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
CONVERSATION_TYPES="${CONVERSATION_TYPES:-simple_description detailed_description}"
CVPR_RELATION_USE_ADALN="${CVPR_RELATION_USE_ADALN:-False}"
CVPR_RELATION_PATCH_GAMMA="${CVPR_RELATION_PATCH_GAMMA:-1.0}"
PYTHONPATH=".:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The released checkpoint used 32 GPUs with batch size 14. Preserve its
# effective batch size when running the public recipe on the default 8 GPUs.
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  requested_world_size="$(( ${NNODES:-1} * ${GPUS_PER_NODE:-${NUM_GPUS:-8}} ))"
  if (( requested_world_size <= 0 || 32 % requested_world_size != 0 )); then
    echo "Set GRADIENT_ACCUMULATION_STEPS explicitly for world size ${requested_world_size}." >&2
    exit 2
  fi
  GRADIENT_ACCUMULATION_STEPS="$((32 / requested_world_size))"
fi
source "${SCRIPT_DIR}/common_torchrun.sh"

export POINTLLM_MULTITASK=1
export POINTLLM_MULTITASK_DATASETS="${POINTLLM_MULTITASK_DATASETS:-mo3d,shape_mating,change_captioning}"
export POINTLLM_FULLMIX_PROBS="${POINTLLM_FULLMIX_PROBS:-0.333,0.333,0.334}"
export POINTLLM_WARMUP_PROBS="${POINTLLM_WARMUP_PROBS:-${POINTLLM_FULLMIX_PROBS}}"
export POINTLLM_MULTITASK_EPOCH_SIZE="${POINTLLM_MULTITASK_EPOCH_SIZE:-180000}"
export POINTLLM_MULTITASK_SEED="${POINTLLM_MULTITASK_SEED:-42}"
export POINTLLM_REINIT_RELATION="${POINTLLM_REINIT_RELATION:-1}"

export POINTLLM_MO3D_DATA_PATH="${POINTLLM_MO3D_DATA_PATH:-${DATA_PATH}}"
export POINTLLM_MO3D_ANNO_PATH="${POINTLLM_MO3D_ANNO_PATH:-data/mo3d/train.json}"
export POINTLLM_SHAPE_MATING_DATA_PATH="${POINTLLM_SHAPE_MATING_DATA_PATH:-${DATA_PATH}}"
export POINTLLM_SHAPE_MATING_ANNO_PATH="${POINTLLM_SHAPE_MATING_ANNO_PATH:-data/shape_mating/train.json}"
export POINTLLM_CHANGE_CAPTIONING_DATA_PATH="${POINTLLM_CHANGE_CAPTIONING_DATA_PATH:-${DATA_PATH}}"
export POINTLLM_CHANGE_CAPTIONING_ANNO_PATH="${POINTLLM_CHANGE_CAPTIONING_ANNO_PATH:-data/change_captioning/train.json}"

read -r -a CONVERSATION_TYPES_ARGS <<< "${CONVERSATION_TYPES}"

CMD=(
  torchrun --nnodes="${NNODES}" --nproc_per_node="${GPUS_PER_NODE}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"
  pointllm/train/train_mem.py
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --anno_path "${POINTLLM_MO3D_ANNO_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --version v1
  --model_max_length 2048
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --evaluation_strategy no
  --save_strategy steps
  --save_steps "${SAVE_STEPS:-5000}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0.
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 1
  --bf16 True
  --fix_llm False
  --fix_pointnet True
  --report_to "${REPORT_TO:-none}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --stage_2 True
  --conversation_types "${CONVERSATION_TYPES_ARGS[@]}"
  --use_color True
  --use_cvpr_model True
  --cvpr_relation_mode patch
  --cvpr_relation_use_adaln "${CVPR_RELATION_USE_ADALN}"
  --cvpr_relation_patch_gamma "${CVPR_RELATION_PATCH_GAMMA}"
)

if [[ -n "${FSDP}" ]]; then
  CMD+=(--fsdp "${FSDP}" --fsdp_transformer_layer_cls_to_wrap "${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}")
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'PYTHONPATH=%q ' "${PYTHONPATH}"
  printf 'POINTLLM_MULTITASK_DATASETS=%q ' "${POINTLLM_MULTITASK_DATASETS}"
  printf 'POINTLLM_FULLMIX_PROBS=%q ' "${POINTLLM_FULLMIX_PROBS}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

export PYTHONPATH
"${CMD[@]}"
