#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
import os
from typing import Optional, List

import torch
import numpy as np

# PyTorch 2.6 compatibility: ensure safe globals are set early.
# NumPy exposes the reconstruct helper under np.core in older releases and
# np._core in newer releases, so collect only attributes that exist.
_safe_globals = [
    np.ndarray,
    np.dtype,
    list,
    tuple,
    dict,
    set,
    frozenset,
    int,
    float,
    str,
    bool,
    bytes,
    type(None),
]
for _root_name in ("core", "_core"):
    _root = getattr(np, _root_name, None)
    _multiarray = getattr(_root, "multiarray", None) if _root is not None else None
    _reconstruct = getattr(_multiarray, "_reconstruct", None) if _multiarray is not None else None
    if _reconstruct is not None and _reconstruct not in _safe_globals:
        _safe_globals.append(_reconstruct)
torch.serialization.add_safe_globals(_safe_globals)

import transformers
from pointllm.train.pointllm_trainer import PointLLMTrainer
from torch.optim import AdamW
from pointllm.train.callbacks import (
    FreezeUnfreezeLLMCallback,
    RelationDeltaLoggerCallback,
    RelationGammaSchedulerCallback,
)


from pointllm import conversation as conversation_lib
from pointllm.model import *
from pointllm.model_cvpr import PointLLMCVPRLlamaForCausalLM, PointLLMCVPRConfig
from pointllm.data import make_object_point_data_module, make_multitask_data_module

# * logger
from pointllm.utils import build_logger

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


def _build_stagewise_param_groups(model, base_lr, relation_scale, llm_init_lr, weight_decay, include_llm, force_include_frozen_llm=False):
    project_params = []
    relation_params = []
    llm_params = []

    for name, param in model.named_parameters():
        if "point_backbone" in name:
            continue  # typically frozen by flag
        
        # ★ stagewise の場合、LLM は freeze されていても optimizer に含める
        is_llm = "point_proj" not in name and "relation_module" not in name and "point_backbone" not in name
        if not param.requires_grad and not (force_include_frozen_llm and is_llm):
            continue
            
        if "point_proj" in name:
            project_params.append(param)
        elif "relation_module" in name:
            relation_params.append(param)
        else:
            llm_params.append(param)

    optim_groups = []
    if project_params:
        optim_groups.append(
            {"params": project_params, "lr": base_lr, "weight_decay": weight_decay, "name": "projector"}
        )
    if relation_params:
        optim_groups.append(
            {
                "params": relation_params,
                "lr": base_lr * relation_scale,
                "weight_decay": weight_decay,
                "name": "relation",
            }
        )
    if include_llm and llm_params:
        optim_groups.append(
            {"params": llm_params, "lr": llm_init_lr, "weight_decay": weight_decay, "name": "llm"}
        )
    return optim_groups


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")
    use_cvpr_model: bool = field(default=False, metadata={"help": "Use CVPR relation module variant."})
    cvpr_use_relation_module: bool = field(default=True, metadata={"help": "Enable/disable relation module (debug isolation)."})
    cvpr_relation_mode: Optional[str] = field(default="object", metadata={"help": "CVPR relation mode: object or patch."})
    cvpr_object_pooling_type: Optional[str] = field(default="mean", metadata={"help": "Pooling type for object mode: mean or max."})
    cvpr_force_object_token_pooling: Optional[bool] = field(default=False, metadata={"help": "Force 1 token per object (max-pool) to reduce sequence length."})
    cvpr_relation_use_adaln: Optional[bool] = field(default=False, metadata={"help": "Enable AdaLN-Zero modulation inside relation module."})
    cvpr_relation_patch_gamma: Optional[float] = field(default=1.0, metadata={"help": "Residual scale for patch-mode relation updates (delta injection strength)."})
    cvpr_stagewise_enable: bool = field(default=False, metadata={"help": "Enable staged relation training features (gamma warmup, partial LLM unfreeze)."})
    cvpr_stagewise_gamma_start: Optional[float] = field(default=None, metadata={"help": "Starting gamma value for patch relation warmup (defaults to cvpr_relation_patch_gamma if unset)."})
    cvpr_stagewise_gamma_end: Optional[float] = field(default=None, metadata={"help": "Final gamma value for patch relation warmup (defaults to cvpr_relation_patch_gamma)."})
    cvpr_stagewise_gamma_warmup_ratio: Optional[float] = field(default=None, metadata={"help": "Ratio of total training steps for gamma warmup."})
    cvpr_stagewise_gamma_warmup_steps: Optional[int] = field(default=None, metadata={"help": "Explicit number of steps for gamma warmup (overrides ratio if set)."})
    cvpr_stagewise_relation_lr_scale: Optional[float] = field(default=None, metadata={"help": "Learning rate multiplier for relation module params."})
    cvpr_stagewise_llm_init_lr: Optional[float] = field(default=None, metadata={"help": "Initial learning rate for LLM params before unfreezing (default 0)."})
    cvpr_stagewise_llm_unfreeze_ratio: Optional[float] = field(default=None, metadata={"help": "Fraction of total steps after which to unfreeze top-k LLM layers."})
    cvpr_stagewise_llm_unfreeze_top_k: Optional[int] = field(default=None, metadata={"help": "Number of top transformer layers to unfreeze."})
    cvpr_stagewise_llm_unfreeze_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate to use for unfrozen LLM layers."})

@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True
    point_identifiers: List[str] = field(default_factory=list, metadata={"help": "A list of point cloud identifier tokens to be added to the tokenizer."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    # Debug / sanity logging
    batch_sanity_log: bool = field(default=False, metadata={"help": "Print per-batch valid label ratio and point token stats."})
    batch_sanity_every_n: int = field(default=1, metadata={"help": "Log every N steps when batch_sanity_log is True."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."}) # * whether to load checkpoints at the mo
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(default=True, metadata={"help": "Whether to fix the PointNet."})

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    # evaluation control (compat with scripts passing --evaluation_strategy)
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "Evaluation strategy: one of 'no', 'steps', 'epoch'"},
    )

    # * for two stage training
    tune_mm_mlp_adapter: bool = field(default=True) # * set True when pre-training, and false when fine-tuning
    stage_2: bool = field(default=False) # * set True when fine-tuning
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None) # * path to the pre-trained projector & output_embed & input_embed
    detatch_point_token: bool = field(default=False) # * deprecated
    # * point backbone ckpt path
    point_backbone_ckpt: str = field(default=None)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ★ シードを最初に設定（モデル初期化前に必須）
    if training_args.seed is not None:
        transformers.set_seed(training_args.seed)
        print(f"🎲 Set random seed to {training_args.seed}")

    training_args.log_level = "info" # * default is passive(warning)
    # * build logger
    logger = build_logger(__name__, training_args.output_dir + '/train.log')
    
    # ★★★ 進捗情報の追加 ★★★
    print("\n" + "="*60)
    print("🚀 PointLLM Training Initialization")
    print("="*60)
    print(f"📁 Model path: {model_args.model_name_or_path}")
    print(f"📁 Output directory: {training_args.output_dir}")
    print(f"🎯 Stage: {'Stage 2 (Fine-tuning)' if training_args.stage_2 else 'Stage 1 (Pre-training)'}")
    print(f"🔧 Fix LLM: {training_args.fix_llm}")
    print(f"🔧 Fix PointNet: {training_args.fix_pointnet}")
    print("="*60)

    print("📥 Loading model...")
    if training_args.model_debug:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        if model_args.use_cvpr_model:
            try:
                if hasattr(model_args, "cvpr_relation_mode"):
                    config.cvpr_relation_mode = model_args.cvpr_relation_mode
                if hasattr(model_args, "cvpr_force_object_token_pooling"):
                    config.cvpr_force_object_token_pooling = bool(model_args.cvpr_force_object_token_pooling)
                if hasattr(model_args, "cvpr_relation_use_adaln"):
                    config.cvpr_relation_use_adaln = bool(getattr(model_args, 'cvpr_relation_use_adaln', False))
                if hasattr(model_args, "cvpr_use_relation_module"):
                    config.cvpr_use_relation_module = bool(model_args.cvpr_use_relation_module)
                if hasattr(model_args, "cvpr_relation_patch_gamma") and getattr(model_args, "cvpr_relation_patch_gamma") is not None:
                    config.cvpr_relation_patch_gamma = float(model_args.cvpr_relation_patch_gamma)
            except Exception as e:
                print(f"[WARN] Failed to preset CVPR config (debug path): {e}")
            model = PointLLMCVPRLlamaForCausalLM._from_config(config)
            print("✅ CVPR model loaded from config (debug mode)")
        else:
            model = PointLLMLlamaForCausalLM._from_config(config)
            print("✅ Model loaded from config (debug mode)")
    else:
        if model_args.use_cvpr_model:
            cvpr_config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
            try:
                cvpr_config.cvpr_relation_mode = model_args.cvpr_relation_mode
                cvpr_config.cvpr_object_pooling_type = getattr(model_args, 'cvpr_object_pooling_type', 'mean')
                cvpr_config.cvpr_force_object_token_pooling = bool(model_args.cvpr_force_object_token_pooling)
                cvpr_config.cvpr_relation_use_adaln = bool(getattr(model_args, 'cvpr_relation_use_adaln', False))
                cvpr_config.cvpr_use_relation_module = bool(model_args.cvpr_use_relation_module)
                if hasattr(model_args, "cvpr_relation_patch_gamma") and getattr(model_args, "cvpr_relation_patch_gamma") is not None:
                    cvpr_config.cvpr_relation_patch_gamma = float(model_args.cvpr_relation_patch_gamma)
            except Exception as e:
                print(f"[WARN] Failed to preset CVPR config: {e}")
            model = PointLLMCVPRLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=cvpr_config,
            )
            print("✅ CVPR model loaded from pretrained checkpoint")
            
            # ★ チェックポイントロード後にRelation Moduleを再初期化（レジューム時はスキップ）
            resuming = any(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
            reinit_relation_env = os.getenv("POINTLLM_REINIT_RELATION", "1").strip().lower()
            reinit_relation = reinit_relation_env not in {"0", "false", "no", "off"}
            if hasattr(model.get_model(), 'relation_module') and model.get_model().relation_module is not None and not resuming and reinit_relation:
                core_model = model.get_model()
                relation_module = core_model.relation_module
                use_adaln = bool(getattr(relation_module, "use_adaln", False))
                encoder = getattr(relation_module, "encoder", None)
                prenorm = False
                if encoder is not None and hasattr(encoder, "layers") and len(encoder.layers) > 0:
                    prenorm = bool(getattr(encoder.layers[0], "prenorm", False))
                logger.info(f"[TRAIN] Re-initializing Relation Module after checkpoint load... (use_adaln={use_adaln}, prenorm={prenorm})")
                relation_module.reset_parameters_for_training()
                logger.info("[TRAIN] Relation Module initialized with trainable identity residual branches.")
                
                # ★★★ ユニットテスト (A): AdaLN なし＝恒等の確認 ★★★
                if not use_adaln:
                    logger.info("[UNITTEST] Testing non-AdaLN identity property...")
                    relation_module.eval()
                    B, T, D = 2, 7, model.config.hidden_size
                    x_test = torch.randn(B, T, D, device=model.device, dtype=model.dtype)
                    with torch.no_grad():
                        y_test = relation_module(x_test)
                    max_diff = (y_test - x_test).abs().max().item()
                    logger.info(f"[UNITTEST] Non-AdaLN identity test: max|y-x|={max_diff:.2e} (expect ~1e-7)")
                    if max_diff < 1e-5:
                        logger.info("[UNITTEST] ✅ PASS: Module behaves as identity (within numerical precision)")
                    else:
                        logger.warning(f"[UNITTEST] ⚠️  WARN: Module deviation {max_diff:.2e} > 1e-5. Check initialization.")
                    relation_module.train()
                
                # ★★★ ユニットテスト (B): AdaLN あり＝ゲート 0 で恒等の確認 ★★★
                if use_adaln:
                    logger.info("[UNITTEST] Testing AdaLN-Zero with zero condition...")
                    relation_module.eval()
                    B, T, D = 2, 7, model.config.hidden_size
                    x_test = torch.randn(B, T, D, device=model.device, dtype=model.dtype)
                    cond_test = torch.zeros(B, D, device=model.device, dtype=model.dtype)
                    with torch.no_grad():
                        y_test = relation_module(x_test, cond=cond_test)
                    max_diff = (y_test - x_test).abs().max().item()
                    logger.info(f"[UNITTEST] AdaLN-Zero identity test (cond=0): max|y-x|={max_diff:.2e} (expect ~1e-7)")
                    if max_diff < 1e-5:
                        logger.info("[UNITTEST] ✅ PASS: AdaLN-Zero behaves as identity with zero condition")
                    else:
                        logger.warning(f"[UNITTEST] ⚠️  WARN: AdaLN-Zero deviation {max_diff:.2e} > 1e-5. Check modulator init.")
                    relation_module.train()
            elif hasattr(model.get_model(), 'relation_module') and model.get_model().relation_module is not None and not reinit_relation:
                logger.info("[TRAIN] Keeping Relation Module weights (POINTLLM_REINIT_RELATION=0).")
        else:
            model = PointLLMLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
            print("✅ Model loaded from pretrained checkpoint")

    model.config.use_cache = False

    stagewise_user = model_args.use_cvpr_model and model_args.cvpr_stagewise_enable
    top_k_cfg = model_args.cvpr_stagewise_llm_unfreeze_top_k
    unfreeze_ratio_cfg = model_args.cvpr_stagewise_llm_unfreeze_ratio
    stagewise_llm_enabled = stagewise_user and (not training_args.fix_llm) and (unfreeze_ratio_cfg is not None)

    if stagewise_user and training_args.fix_llm and unfreeze_ratio_cfg is not None:
        logger.info("[STAGEWISE] fix_llm=True のため LLM 制御オプションは無効化されます。")

    # ★ stagewise_llm_enabled の場合は、最初から LLM を freeze する
    if training_args.fix_llm or stagewise_llm_enabled:
        if training_args.fix_llm:
            logger.info("LLM is fixed. Fix_llm flag is set to True")
        else:
            logger.info("LLM is initially frozen for stagewise training (will be unfrozen later)")
        # * fix llama, lm_head, pointnet, projection layer here
        model.requires_grad_(False)
        core = model.get_model()
        core.fix_llm = True  # Temporarily set to True for initial freeze
        # relation と projector は学習対象に戻す
        core.point_proj.requires_grad_(True)
        if getattr(core, "relation_module", None) is not None:
            core.relation_module.requires_grad_(True)
        # point_backbone の requires_grad は fix_pointnet ブロックで統一管理
    else:
        core = model.get_model()
        core.fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    print("🔤 Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    print("✅ Tokenizer loaded")

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if not training_args.fix_pointnet:
        # * not fix pointnet
        logger.info("Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded.")
        model.get_model().fix_pointnet = False
    else:
        logger.info("Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded.")
        model.get_model().fix_pointnet = True # * use with torch.inference_mode to control, not requires_grad for fsdp for second stage
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            model.get_model().point_backbone.requires_grad_(False) # * fix pointnet for first stage, need for fsdp in stage2
    
    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_point_backbone_config
        logger.info("Point projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    if not training_args.stage_2:
        # * we assume in stage2, llm, point_backbone, and projection layer can be loaded from the model checkpoint
        print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)
        model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm, data_args=data_args)
    else:
        # * stage2
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 

    point_backbone_config = model.get_model().point_backbone_config

    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    print("📊 Creating data module...")
    
    # Check if multitask mode is enabled via environment variable
    multitask_enabled = os.getenv('POINTLLM_MULTITASK', '0') == '1'
    
    if multitask_enabled:
        print("🔥 Multi-task mode enabled!")
        
        # Parse dataset names from environment
        dataset_names = [
            name.strip()
            for name in os.getenv('POINTLLM_MULTITASK_DATASETS', 'mo3d,shape_easy,change').split(',')
            if name.strip()
        ]
        
        # Build dataset configs with paths
        dataset_configs = []
        for ds_name in dataset_names:
            data_path_key = f'POINTLLM_{ds_name.upper()}_DATA_PATH'
            anno_path_key = f'POINTLLM_{ds_name.upper()}_ANNO_PATH'
            dataset_configs.append({
                'name': ds_name,
                'data_path': os.getenv(data_path_key, ''),
                'anno_path': os.getenv(anno_path_key, ''),
            })
        
        # Create multitask configuration
        warmup_probs = [
            float(x.strip())
            for x in os.getenv('POINTLLM_WARMUP_PROBS', '0.7,0.15,0.15').split(',')
            if x.strip()
        ]
        fullmix_probs = [
            float(x.strip())
            for x in os.getenv('POINTLLM_FULLMIX_PROBS', '0.362,0.304,0.335').split(',')
            if x.strip()
        ]
        if fullmix_probs and len(fullmix_probs) != len(dataset_configs):
            raise ValueError(
                "POINTLLM_FULLMIX_PROBS length must match POINTLLM_MULTITASK_DATASETS "
                f"({len(fullmix_probs)} probs for {len(dataset_configs)} datasets)."
            )
        if warmup_probs and len(warmup_probs) != len(dataset_configs):
            raise ValueError(
                "POINTLLM_WARMUP_PROBS length must match POINTLLM_MULTITASK_DATASETS "
                f"({len(warmup_probs)} probs for {len(dataset_configs)} datasets)."
            )

        multitask_config = {
            'datasets': dataset_configs,  # Now a list of dicts
            'alpha': float(os.getenv('POINTLLM_MULTITASK_ALPHA', '0.5')),
            'warmup_ratio': float(os.getenv('POINTLLM_MULTITASK_WARMUP_RATIO', '0.15')),
            'warmup_probs': warmup_probs,
            'fullmix_probs': fullmix_probs,
            'epoch_size': int(os.getenv('POINTLLM_MULTITASK_EPOCH_SIZE', '0')),
            'seed': int(os.getenv('POINTLLM_MULTITASK_SEED', str(training_args.seed))),
            'phase_aware': False,
        }
        
        print(f"  Datasets: {[ds['name'] for ds in dataset_configs]}")
        print(f"  Alpha (temperature): {multitask_config['alpha']}")
        print(f"  Warmup ratio: {multitask_config['warmup_ratio']}")
        print(f"  Warmup probs: {multitask_config['warmup_probs']}")
        print(f"  Full-mix probs: {multitask_config['fullmix_probs']}")
        print(f"  Virtual epoch size: {multitask_config['epoch_size'] or 'sum(dataset sizes)'}")
        
        # Add multitask_config to data_args
        data_args.multitask_config = multitask_config
        data_args.pointnum = getattr(data_args, 'pointnum', 8192)
        data_args.use_color = getattr(data_args, 'use_color', True)
        
        data_module = make_multitask_data_module(tokenizer=tokenizer, data_args=data_args)
    else:
        print("📦 Single-task mode")
        data_module = make_object_point_data_module(tokenizer=tokenizer, data_args=data_args)
    
    print("✅ Data module created")

    # [SANITY CHECK] Verify valid labels in first batch (debug for loss=0 issue)
    # ★ コメントアウト：データセットの状態を変更する可能性があるため無効化
    try:
        from torch.utils.data import DataLoader
        print("🔍 [SANITY] Checking first batch for valid labels...")
        tmp_loader = DataLoader(
            data_module["train_dataset"], 
            batch_size=1, 
            shuffle=False, 
            collate_fn=data_module["data_collator"]
        )
        batch = next(iter(tmp_loader))
        valid_labels = int((batch["labels"] != -100).sum().item())
        total_labels = batch["labels"].numel()
        print(f"✅ [SANITY] Valid labels in first batch: {valid_labels}/{total_labels} ({100*valid_labels/total_labels:.1f}%)")
        if valid_labels == 0:
            print("⚠️  [SANITY] WARNING: No valid labels found! All labels are -100. Loss will be 0!")
            print(f"    Input shape: {batch['input_ids'].shape}")
            if 'attention_mask' in batch:
                print(f"    Attention mask shape: {batch['attention_mask'].shape}")
            print(f"    Labels shape: {batch['labels'].shape}")
    except Exception as e:
        print(f"⚠️  [SANITY] Could not check batch: {e}")

    callbacks = []
    optimizers = None

    # ★ デバッグ：Stagewise 設定サマリー（rank 0のみ）
    if stagewise_user:
        import torch.distributed as dist
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            print("\n" + "="*60)
            print("🎛️  [Stagewise Training] Configuration Summary")
            print("="*60)
            print(f"🔧 CVPR Stagewise: Enabled")
            print(f"🔧 Relation module: {'Enabled' if model_args.cvpr_use_relation_module else 'Disabled'}")
            print(f"🔧 Relation mode: {model_args.cvpr_relation_mode}")
            if model_args.cvpr_relation_mode == "patch":
                print(f"   └─ Gamma (patch residual scale): {model_args.cvpr_relation_patch_gamma}")
                if model_args.cvpr_stagewise_gamma_start is not None or model_args.cvpr_stagewise_gamma_warmup_ratio is not None:
                    print(f"   └─ Gamma warmup: {model_args.cvpr_stagewise_gamma_start} → {model_args.cvpr_stagewise_gamma_end}")
                    print(f"      (ratio: {model_args.cvpr_stagewise_gamma_warmup_ratio}, steps: {model_args.cvpr_stagewise_gamma_warmup_steps})")
            if model_args.cvpr_stagewise_relation_lr_scale is not None and model_args.cvpr_stagewise_relation_lr_scale != 1.0:
                print(f"📈 Relation LR scale: {model_args.cvpr_stagewise_relation_lr_scale}x")
            if stagewise_llm_enabled:
                print(f"🔓 LLM unfreeze: Step {unfreeze_ratio_cfg*100:.0f}% (ratio={unfreeze_ratio_cfg})")
                print(f"   └─ Unfreeze layers: {'ALL' if top_k_cfg is None or top_k_cfg <= 0 else f'top-{top_k_cfg}'}")
                print(f"   └─ Target LR: {model_args.cvpr_stagewise_llm_unfreeze_lr or training_args.learning_rate}")
            else:
                print(f"🔒 LLM: {'Fixed (fix_llm=True)' if training_args.fix_llm else 'Fully trainable (fix_llm=False)'}")
            print("="*60 + "\n")

    if stagewise_user:
        callbacks.append(RelationDeltaLoggerCallback())

        default_gamma = model_args.cvpr_relation_patch_gamma if model_args.cvpr_relation_patch_gamma is not None else 1.0
        end_gamma = float(model_args.cvpr_stagewise_gamma_end) if model_args.cvpr_stagewise_gamma_end is not None else float(default_gamma)
        start_gamma = float(model_args.cvpr_stagewise_gamma_start) if model_args.cvpr_stagewise_gamma_start is not None else (0.2 if end_gamma > 0 else end_gamma)
        core_model = model.get_model()
        core_model.set_relation_patch_gamma(start_gamma)

        warmup_steps = model_args.cvpr_stagewise_gamma_warmup_steps
        warmup_ratio = model_args.cvpr_stagewise_gamma_warmup_ratio
        if warmup_steps is None and warmup_ratio is None and start_gamma != end_gamma:
            warmup_ratio = 0.3
        if start_gamma != end_gamma or warmup_steps or warmup_ratio:
            callbacks.append(
                RelationGammaSchedulerCallback(
                    start=start_gamma,
                    end=end_gamma,
                    warmup_steps=warmup_steps,
                    warmup_ratio=warmup_ratio,
                )
            )

        relation_scale = float(model_args.cvpr_stagewise_relation_lr_scale) if model_args.cvpr_stagewise_relation_lr_scale is not None else 1.0
        need_custom_optimizer = stagewise_llm_enabled or relation_scale != 1.0

        if need_custom_optimizer:
            llm_init_lr = float(model_args.cvpr_stagewise_llm_init_lr) if (stagewise_llm_enabled and model_args.cvpr_stagewise_llm_init_lr is not None) else 0.0
            param_groups = _build_stagewise_param_groups(
                model=model,
                base_lr=training_args.learning_rate,
                relation_scale=relation_scale,
                llm_init_lr=llm_init_lr,
                weight_decay=training_args.weight_decay,
                include_llm=stagewise_llm_enabled,
                force_include_frozen_llm=stagewise_llm_enabled,  # ★ freeze された LLM も optimizer に含める
            )
            if param_groups:
                optimizers = (
                    AdamW(
                        param_groups,
                        betas=(training_args.adam_beta1, training_args.adam_beta2),
                        eps=training_args.adam_epsilon,
                    ),
                    None,
                )

        if stagewise_llm_enabled:
            # top_k_cfg が None の場合は 0 (全層 unfreeze) とする
            top_k_layers = int(top_k_cfg) if top_k_cfg is not None else 0
            unfreeze_ratio = float(unfreeze_ratio_cfg)
            target_lr = model_args.cvpr_stagewise_llm_unfreeze_lr
            if target_lr is None:
                target_lr = training_args.learning_rate
            callbacks.append(
                FreezeUnfreezeLLMCallback(
                    unfreeze_ratio=unfreeze_ratio,
                    top_k_layers=top_k_layers,
                    target_lr=float(target_lr),
                )
            )
        elif top_k_cfg is not None or unfreeze_ratio_cfg is not None:
            logger.info("[STAGEWISE] LLM 段階的アンロック条件が満たされないため、LLM 制御をスキップします。")

    print("🏃 Initializing trainer...")
    
    # ★ FSDP環境かどうかをチェック
    fsdp_enabled = training_args.fsdp is not None and len(training_args.fsdp) > 0
    
    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer_kwargs.update(data_module)
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    
    # ★ FSDP環境では optimizers を渡せないので、param_groups を後で設定
    if optimizers is not None and not fsdp_enabled:
        trainer_kwargs["optimizers"] = optimizers

    trainer = PointLLMTrainer(**trainer_kwargs)
    
    # ★ FSDP環境の場合、param_groups を trainer の属性として設定
    if fsdp_enabled and optimizers is not None:
        import torch.distributed as dist
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        # Extract param groups from the optimizer tuple
        if isinstance(optimizers, tuple) and len(optimizers) > 0:
            opt = optimizers[0]
            if hasattr(opt, 'param_groups'):
                trainer.stagewise_param_groups = opt.param_groups
                if is_main_process:
                    print(f"✅ [FSDP] Stagewise param groups stored ({len(opt.param_groups)} groups)")
            else:
                trainer.stagewise_param_groups = None
        else:
            trainer.stagewise_param_groups = None
    else:
        trainer.stagewise_param_groups = None
    print("✅ Trainer initialized")

    print("\n" + "="*60)
    print("🎯 Starting Training Process")
    print("="*60)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("📍 Resuming from existing checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("🆕 Starting training from scratch...")
        trainer.train()
    
    print("💾 Saving final model state...")
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)
    print("🎉 Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    train()
