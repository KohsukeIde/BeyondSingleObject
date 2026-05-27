#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: TASK=mo3d|shape_mating|change_captioning $0 /path/to/inference.json" >&2
  exit 2
fi

INPUT_FILE="$1"
TASK="${TASK:-mo3d}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/llm_eval/${TASK}_llm_eval.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-10}"
ANNOTATION="${ANNOTATION:-}"

case "${TASK}" in
  mo3d)
    CMD=(python -m pointllm.eval.cvpr.evaluate_3d_qa_fast "${INPUT_FILE}" --output "${OUTPUT_FILE}" --batch_size "${BATCH_SIZE}")
    ;;
  shape_mating|change_captioning)
    CMD=(python -m pointllm.eval.cvpr.evaluate_change_captioning "${INPUT_FILE}" --output "${OUTPUT_FILE}" --batch_size "${BATCH_SIZE}")
    if [[ -n "${ANNOTATION}" ]]; then
      CMD+=(--annotation "${ANNOTATION}")
    fi
    ;;
  *)
    echo "Unknown TASK=${TASK}. Use mo3d, shape_mating, or change_captioning." >&2
    exit 2
    ;;
esac

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"
"${CMD[@]}"
