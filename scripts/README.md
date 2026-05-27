# Scripts

Public entrypoints are shell wrappers around the training and evaluation
modules. They are designed to run locally or under `torchrun` without
cluster-specific job files.

## Training

```bash
DRY_RUN=1 scripts/train/train_joint.sh
```

`scripts/train/train_joint.sh` is the public fine-tuning recipe for MO3D,
Shape Mating, and Change Captioning. It can also read PointLLM-format
caption/instruction JSONs to preserve the upstream single-object interface.

Common overrides:

```bash
MODEL_PATH=checkpoints/pointllm-stage1
DATA_PATH=data/point_clouds
OUTPUT_DIR=outputs/joint
NNODES=1
GPUS_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=29510
PER_DEVICE_TRAIN_BATCH_SIZE=14
GRADIENT_ACCUMULATION_STEPS=1
FSDP="full_shard auto_wrap"
NUM_TRAIN_EPOCHS=2
LEARNING_RATE=1.5e-5
```

Dataset-path overrides:

```bash
POINTLLM_POINTLLM_CAPTION_ANNO_PATH=data/pointllm/PointLLM_brief_description_660K_filtered.json
POINTLLM_POINTLLM_INSTRUCTION_ANNO_PATH=data/pointllm/PointLLM_complex_instruction_70K.json
POINTLLM_POINTLLM_MULTI_INSTRUCTION_ANNO_PATH=data/pointllm/complex_instruction_stage2_multi_pc_70K_gpt.json
POINTLLM_MO3D_ANNO_PATH=data/mo3d/train.json
POINTLLM_SHAPE_MATING_ANNO_PATH=data/shape_mating/train.json
POINTLLM_CHANGE_CAPTIONING_ANNO_PATH=data/change_captioning/train.json
```

For the public Shape Mating benchmark, `data/shape_mating/train.json` should
point to the released Thingi10K split.

## Evaluation

Inference:

```bash
DRY_RUN=1 scripts/eval/infer.sh
```

LLM-based paper metrics:

```bash
TASK=mo3d scripts/eval/eval_llm.sh outputs/mo3d_eval/inference.json
TASK=shape_mating ANNOTATION=data/shape_mating/test.json scripts/eval/eval_llm.sh outputs/shape_mating_eval/inference.json
TASK=change_captioning ANNOTATION=data/change_captioning/eval_subset.json scripts/eval/eval_llm.sh outputs/change_captioning_eval_subset/inference.json
```

For Change Captioning inference, set `SCORE_VERIFY_OPTIONS=1` so the binary
verification turn is scored as a constrained Yes/No decision.

Supplemental NLP overlap metrics:

```bash
TASK=mo3d scripts/eval/eval_nlp.sh outputs/mo3d_eval/inference.json
TASK=shape_mating ANNO_PATH=data/shape_mating/test.json scripts/eval/eval_nlp.sh outputs/shape_mating_eval/inference.json
TASK=change_captioning scripts/eval/eval_nlp.sh outputs/change_captioning_eval_subset/inference.json
```

ModelNet40 CLIP classification:

```bash
MODEL_PATH=checkpoints/multi-3dllm-classification scripts/eval/eval_modelnet.sh
```
