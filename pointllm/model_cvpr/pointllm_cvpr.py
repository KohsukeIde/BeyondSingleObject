from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import os
import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext

logger = logging.getLogger(__name__)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from pointllm.model.pointllm import PointLLMLlamaModel, PointLLMConfig
from pointllm.utils import cfg_from_yaml_file
from pointllm.model.pointbert.point_encoder import PointTransformer

from .relation import SimpleRelationModule
from torch.nn import functional as F


def _log_feat_stats(tag: str, feats):
    """特徴量の統計情報を出力する関数（デバッグ用）"""
    debug_env = os.getenv("POINTLLM_DEBUG_FEATS", "0")
    if debug_env != "1":
        return
    try:
        if isinstance(feats, list):
            tensors = []
            for item in feats:
                if isinstance(item, list):
                    tensors.extend(item)
                else:
                    tensors.append(item)
            if not tensors:
                print(f"[FEATDBG] {tag}: {{'empty': True}}", flush=True)
                return
            cat = torch.cat([t.detach() for t in tensors], dim=0)
        else:
            cat = feats.detach()
        stats = {
            "shape": tuple(cat.shape),
            "dtype": str(cat.dtype),
            "device": str(cat.device),
            "mean": float(cat.mean().item()),
            "std": float(cat.std().item()),
            "min": float(cat.min().item()),
            "max": float(cat.max().item()),
        }
        print(f"[FEATDBG] {tag}: {stats}", flush=True)
    except Exception as exc:
        print(f"[FEATDBG] {tag} err: {exc}", flush=True)


def token_fps_pool(tokens: torch.Tensor, M: int = 32, metric: str = "cos") -> torch.Tensor:
    """Token-level FPS pooling: compress T tokens to M representatives.
    
    Note: @torch.no_grad() removed to allow gradients to flow through for training.
    The pooling operation is differentiable via the cluster averaging step.
    """
    T, D = tokens.shape
    if T <= M:
        return tokens
    
    X = F.normalize(tokens, p=2, dim=-1) if metric == "cos" else tokens
    dists = torch.full((T,), float("inf"), device=tokens.device, dtype=tokens.dtype)
    first = torch.argmax((X * X).sum(dim=-1))
    centers = [first.item()]
    last = X[first]
    
    for _ in range(M - 1):
        new_d = 1.0 - (X @ last) if metric == "cos" else (X - last).pow(2).sum(-1)
        dists = torch.minimum(dists, new_d)
        nxt = torch.argmax(dists)
        centers.append(nxt.item())
        last = X[nxt]
    
    C_idx = torch.as_tensor(centers, device=tokens.device)
    C_raw = tokens.index_select(0, C_idx)  # (M,D)
    
    # Assign each token to nearest representative
    if metric == "cos":
        sim = X @ X.index_select(0, C_idx).t()
        assign = torch.argmax(sim, dim=-1)
    else:
        c = X.index_select(0, C_idx)
        x2 = (X * X).sum(-1, keepdim=True)
        c2 = (c * c).sum(-1).unsqueeze(0)
        dist2 = x2 + c2 - 2 * (X @ c.t())
        assign = torch.argmin(dist2, dim=-1)
    
    # Cluster average (empty clusters use representative itself)
    out = []
    for m in range(M):
        mask = (assign == m)
        out.append(tokens[mask].mean(0, keepdim=True) if mask.any() else C_raw[m:m+1])
    return torch.cat(out, dim=0)


class PointLLMCVPRConfig(PointLLMConfig):
    """Extends the default PointLLM config with relation module switches."""

    model_type = "pointllm_cvpr"
    cvpr_use_relation_module: bool = True
    cvpr_relation_use_adaln: bool = True
    cvpr_relation_num_layers: int = 1
    cvpr_relation_num_heads: int = 8
    cvpr_relation_dropout: float = 0.1
    cvpr_force_object_token_pooling: bool = False
    cvpr_relation_mode: str = "object"  # "object" | "patch" | "micro" | "fast_patch"
    cvpr_object_pooling_type: str = "mean"  # "mean" | "max" for object mode
    cvpr_token_budget_per_obj: int = 32
    cvpr_relation_patch_gamma: float = 1.0


class PointLLMCVPRLlamaModel(PointLLMLlamaModel):
    config_class = PointLLMCVPRConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # Log pooling status without rebuilding (preserve checkpoint shape)
        logger.info(
            f"[CVPR] Pooling status: use_max_pool(backbone)={self.point_backbone_config.get('use_max_pool', False)}, "
            f"point_token_len={self.point_backbone_config.get('point_token_len')}, "
            f"relation_mode={getattr(config, 'cvpr_relation_mode', 'object')}, "
            f"object_pooling_type={getattr(config, 'cvpr_object_pooling_type', 'mean')}"
        )

        hidden_size = self.point_backbone_config["project_output_dim"]
        use_relation = getattr(config, "cvpr_use_relation_module", True)
        self.relation_use_adaln = bool(getattr(config, "cvpr_relation_use_adaln", True)) if use_relation else False
        if use_relation:
            self.relation_module = SimpleRelationModule(
                hidden_size=hidden_size,
                num_layers=getattr(config, "cvpr_relation_num_layers", 1),
                num_heads=getattr(config, "cvpr_relation_num_heads", 8),
                dropout=getattr(config, "cvpr_relation_dropout", 0.1),
                use_adaln=self.relation_use_adaln,
            )
        else:
            self.relation_module = None
        
        # [IMPROVEMENT 2] Condition LayerNorm for stabilization (especially in bf16)
        if self.relation_use_adaln:
            self.cond_ln = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.cond_ln = None
        
        # Note: condition_projector is NOT needed because:
        # - Relation Module operates in 4096-dim space (after Point Projector)
        # - Question embeddings are also 4096-dim
        # - Both are in the same space, no projection needed!

        # relation mode and micro-token budget
        self.relation_mode = getattr(config, "cvpr_relation_mode", "object")
        self._micro_M = getattr(config, "cvpr_token_budget_per_obj", 32)
        gamma = float(getattr(config, "cvpr_relation_patch_gamma", 1.0))
        self.relation_gamma = gamma
        self._relation_patch_gamma = gamma  # legacy name for backward compatibility

        # fast_patch mode: adjust point_token_len to M (LLM will receive M tokens per object)
        if self.relation_mode.lower() == "fast_patch":
            prev_len = self.point_backbone_config.get("point_token_len")
            self.point_backbone_config["point_token_len"] = int(self._micro_M)
            logger.info(f"[CVPR] fast_patch: point_token_len {prev_len} -> {self._micro_M}")

    def _apply_relation_object(self, features: List[torch.Tensor], cond: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Object-level relation like Chat-3D, even if each object has multiple tokens.

        Steps per sample:
        1) Pool each object's token set -> one vector per object (mean or max).
        2) Run TransformerEncoder over object tokens (set-wise self-attention).
        3) Broadcast the object-level update back to all tokens of that object as a residual.
        """
        if self.relation_module is None or len(features) == 0:
            return features

        device = features[0].device
        pooling_type = getattr(self.config, "cvpr_object_pooling_type", "mean")
        obj_vecs = []
        lengths = []
        for feat in features:  # feat: (Ti, D)
            lengths.append(feat.shape[0])
            if pooling_type == "max":
                obj_vecs.append(feat.max(dim=0, keepdim=True)[0])  # (1, D)
            else:  # default: mean
                obj_vecs.append(feat.mean(dim=0, keepdim=True))  # (1, D)
        obj_tokens = torch.cat(obj_vecs, dim=0).unsqueeze(0).to(device)  # (1, M, D)

        cond_tensor = None
        if self.relation_use_adaln and cond is not None:
            cond_tensor = cond.to(device)

        _log_feat_stats("relation_in(object)", [obj_tokens.squeeze(0)])
        obj_updated = self.relation_module(obj_tokens, cond=cond_tensor).squeeze(0)  # (M, D)
        _log_feat_stats("relation_out(object)", [obj_updated])
        obj_orig = obj_tokens.squeeze(0)  # (M, D)
        delta = obj_updated - obj_orig  # (M, D)

        updated_features: List[torch.Tensor] = []
        for i, feat in enumerate(features):
            Ti = feat.shape[0]
            upd = feat + delta[i].unsqueeze(0).expand(Ti, -1)
            updated_features.append(upd)
        return updated_features

    def set_relation_patch_gamma(self, value: float) -> None:
        """Allow schedulers to adjust the patch-level residual scale during training."""
        try:
            val = float(value)
        except Exception:
            val = 1.0
        self.relation_gamma = val
        self._relation_patch_gamma = val

    def _apply_relation_patch(self, features: List[torch.Tensor], cond: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Patch-concatenation relation across all patch tokens (inter- and intra-object attention).

        Concatenate all patch tokens, run the relation module, and split back per object.
        """
        if self.relation_module is None or len(features) == 0:
            return features
        lengths = [feat.shape[0] for feat in features]
        total_tokens = sum(lengths)
        if total_tokens == 0:
            return features
        device = features[0].device
        base = torch.cat(features, dim=0)  # (sum(Ti), D)
        cat = base.unsqueeze(0)  # (1, sum(Ti), D)
        cond_tensor = None
        if self.relation_use_adaln and cond is not None:
            cond_tensor = cond.to(device)
            # [IMPROVEMENT 2] Stabilize cond with LayerNorm (especially for bf16)
            if self.cond_ln is not None:
                cond_tensor = self.cond_ln(cond_tensor)
        _log_feat_stats("relation_in(patch)", [base])
        updated = self.relation_module(cat, cond=cond_tensor).squeeze(0)  # (sum(Ti), D)
        _log_feat_stats("relation_out(patch)", [updated])
        
        # Compute delta
        delta = updated - base
        
        # [IMPROVEMENT 1] RMS clipping to prevent delta explosion (cap=1.0)
        # This prevents the relation module from disrupting LLM's language optimization
        delta_rms = delta.pow(2).mean().sqrt()
        delta_cap = getattr(self.config, "cvpr_relation_delta_cap", 1.0)
        if delta_rms > delta_cap:
            delta = delta * (delta_cap / (delta_rms + 1e-6))
            if os.getenv("POINTLLM_RELATION_DEBUG"):
                logger.debug(f"[RELATION] Delta RMS clipped: {delta_rms:.4f} → {delta_cap:.4f}")
        
        # Apply gamma blending
        gamma = float(getattr(self, "relation_gamma", getattr(self, "_relation_patch_gamma", 1.0)))
        if gamma != 1.0:
            delta = delta * gamma
        patched = base + delta

        out: List[torch.Tensor] = []
        offset = 0
        for L in lengths:
            out.append(patched[offset:offset + L])
            offset += L
        return out

    def _apply_relation_micro(self, features: List[torch.Tensor], cond: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Micro-token relation with patch-wise delta (A: micro mode).
        
        Steps:
        1) Compress each object's tokens (Ti,D) -> micro tokens (M,D) via token_fps_pool
        2) Run relation module on concatenated micro tokens
        3) For each object, compute patch-wise delta via cross-attention (patch -> micro)
        """
        M = getattr(self.config, "cvpr_token_budget_per_obj", 32)
        if self.relation_module is None or len(features) == 0:
            return features
        
        pooled = [token_fps_pool(feat, M=M, metric="cos") for feat in features]  # each: (M,D)
        
        cat = torch.cat(pooled, dim=0).unsqueeze(0)  # (1, sum(Mi), D)
        cond_tensor = None
        if self.relation_use_adaln and cond is not None:
            cond_tensor = cond.to(cat.device)
        _log_feat_stats("relation_in(micro)", [cat.squeeze(0)])
        cat_upd = self.relation_module(cat, cond=cond_tensor).squeeze(0)  # (sum(Mi), D)
        _log_feat_stats("relation_out(micro)", [cat_upd])
        
        upd_list = []
        ptr = 0
        for P in pooled:
            upd_list.append(cat_upd[ptr:ptr + P.size(0)])
            ptr += P.size(0)
        
        updated = []
        for feat, P_upd in zip(features, upd_list):  # feat:(T,D), P_upd:(M,D)
            D = feat.size(-1)
            A = torch.softmax((feat @ P_upd.t()) / (D ** 0.5), dim=-1)  # (T,M)
            delta = A @ P_upd  # (T,D)
            updated.append(feat + delta)
        return updated

    @staticmethod
    def _find_last_subsequence(sequence: torch.Tensor, pattern: Optional[Sequence[int]]) -> Optional[int]:
        if pattern is None or len(pattern) == 0:
            return None
        if sequence.dim() != 1:
            sequence = sequence.view(-1)
        seq_list = sequence.tolist()
        pat_list = [int(p) for p in pattern]
        plen = len(pat_list)
        if plen == 0 or len(seq_list) < plen:
            return None
        for idx in range(len(seq_list) - plen, -1, -1):
            if seq_list[idx:idx + plen] == pat_list:
                return idx
        return None

    def _extract_condition_vector(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        point_backbone_config: dict,
    ) -> Optional[torch.Tensor]:
        if input_ids is None or input_embeds is None:
            return None

        if input_ids.dim() != 1:
            input_ids = input_ids.view(-1)

        seq_len = input_ids.size(0)
        if seq_len == 0:
            return None

        # Try all tokenization variants for robust matching
        assistant_start = None
        for key in ("assistant_token_id_variants", "assistant_token_ids"):
            ids = point_backbone_config.get(key, None)
            if ids:
                if isinstance(ids[0], (list, tuple)):  # variants (list of lists)
                    for pattern in ids:
                        assistant_start = self._find_last_subsequence(input_ids, pattern)
                        if assistant_start is not None:
                            break
                else:  # single pattern (list of ints)
                    assistant_start = self._find_last_subsequence(input_ids, ids)
                if assistant_start is not None:
                    break
        
        if assistant_start is None:
            # CRITICAL: If ASSISTANT: token not found during training, this is a critical error.
            # We MUST stop to prevent training with leaked answer information.
            # During inference, we return None to trigger zero-condition fallback.
            if self.training:
                assistant_ids = point_backbone_config.get("assistant_token_ids", [])
                decoded_seq = self.config.tokenizer.decode(input_ids) if hasattr(self.config, 'tokenizer') else "<cannot decode>"
                raise RuntimeError(
                    f"[AdaLN Training] ASSISTANT: token sequence not found in input!\n"
                    f"  assistant_token_ids (representative): {assistant_ids}\n"
                    f"  input_ids length: {len(input_ids)}\n"
                    f"  Decoded (first 200 chars): {decoded_seq[:200]}...\n"
                    f"This is a critical error during training. Check:\n"
                    f"  1. _record_conversation_token_ids is correctly setting assistant_token_ids\n"
                    f"  2. Conversation template matches the tokenization\n"
                    f"  3. Training data has proper conversation format"
                )
            else:
                # During inference, return None to trigger zero-condition fallback
                if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
                    import logging
                    logger = logging.getLogger(__name__)
                    assistant_ids = point_backbone_config.get("assistant_token_ids", [])
                    logger.warning(
                        f"[AdaLN Inference] ASSISTANT: token not found. Will use zero-condition fallback. "
                        f"assistant_token_ids={assistant_ids}"
                    )
                return None
        assistant_start = max(0, min(int(assistant_start), seq_len))

        search_slice = input_ids[:assistant_start]
        question_start = 0

        mm_use_point_start_end = point_backbone_config.get("mm_use_point_start_end", True)
        point_end_token = point_backbone_config.get("point_end_token")
        if mm_use_point_start_end and point_end_token is not None and assistant_start > 0:
            matches = (search_slice == point_end_token).nonzero(as_tuple=False)
            if matches.numel() > 0:
                question_start = int(matches[-1].item()) + 1

        if question_start == 0:
            point_patch_token = point_backbone_config.get("point_patch_token")
            if point_patch_token is not None and assistant_start > 0:
                matches = (search_slice == point_patch_token).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    question_start = int(matches[-1].item()) + 1

        if question_start == 0:
            user_ids = point_backbone_config.get("user_token_ids")
            user_start = self._find_last_subsequence(search_slice, user_ids)
            if user_start is not None:
                question_start = user_start + len(user_ids)

        question_start = max(0, min(question_start, assistant_start))

        ignore_ids = {
            token_id
            for token_id in (
                point_backbone_config.get("point_patch_token"),
                point_backbone_config.get("point_start_token"),
                point_backbone_config.get("point_end_token"),
                self.config.pad_token_id,
                self.config.bos_token_id,
                self.config.eos_token_id,
            )
            if token_id is not None
        }

        candidate_indices: List[int] = []
        for idx in range(question_start, assistant_start):
            token_id = int(input_ids[idx].item())
            if token_id in ignore_ids:
                continue
            candidate_indices.append(idx)

        if not candidate_indices and assistant_start > question_start:
            candidate_indices = list(range(question_start, assistant_start))

        if not candidate_indices:
            hidden_dim = input_embeds.size(-1)
            raw_cond = input_embeds.new_zeros(1, hidden_dim)
        else:
            cond_stack = torch.stack([input_embeds[idx] for idx in candidate_indices], dim=0)
            raw_cond = cond_stack.mean(dim=0, keepdim=True)
        
        # ★★★ DEBUG ★★★
        if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
            print(f"[ADALN_DEBUG] _extract_condition_vector:")
            print(f"  raw_cond shape: {raw_cond.shape}")
            print(f"  raw_cond mean: {raw_cond.mean().item():.6f}")
            print(f"  raw_cond std: {raw_cond.std().item():.6f}")
            print(f"  raw_cond min: {raw_cond.min().item():.6f}")
            print(f"  raw_cond max: {raw_cond.max().item():.6f}")
            print(f"  raw_cond has NaN: {raw_cond.isnan().any().item()}")
            print(f"  No projection needed (already in LLM space: 4096-dim)")
            print()
        
        # Return as-is: both question embeddings and relation features are 4096-dim
        return raw_cond

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        num_point_clouds_valid: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # デバッグ用: forward()呼び出しを追跡
        if os.getenv("POINTLLM_DEBUG_FORWARD", "0") == "1":
            input_shape = tuple(input_ids.shape) if input_ids is not None else "None"
            pc_status = "Present" if point_clouds is not None else "None"
            past_status = "Present" if past_key_values is not None else "None"
            print(f"[FWD] input_ids={input_shape}, point_clouds={pc_status}, past_key_values={past_status}", flush=True)

        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, "point_backbone", None)
        point_backbone_config = getattr(self, "point_backbone_config", None)

        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    point_backbone.eval()

                if isinstance(point_clouds, list):
                    point_features: list[list[torch.Tensor]] = []
                    for sample in point_clouds:
                        if isinstance(sample, list):
                            sample_feats = [point_backbone(pc.unsqueeze(0))[0] for pc in sample]
                        else:
                            sample_feats = [point_backbone(sample.unsqueeze(0))[0]]
                        point_features.append(sample_feats)
                else:
                    if point_clouds.dim() == 4:
                        batch_size, num_clouds, _, _ = point_clouds.shape
                        point_features = []
                        for b in range(batch_size):
                            feats = [point_backbone(point_clouds[b, idx].unsqueeze(0))[0] for idx in range(num_clouds)]
                            point_features.append(feats)
                    else:
                        feats = point_backbone(point_clouds)
                        point_features = [[feat] for feat in feats]

            _log_feat_stats("backbone_out", point_features)
            
            projected_features: list[list[torch.Tensor]] = []
            for sample_feats in point_features:
                if self.point_backbone_config["projection_hidden_layer"] > 0:
                    projected = [self.point_proj(feat) for feat in sample_feats]
                else:
                    projected = [self.point_proj(feat) for feat in sample_feats]
                projected_features.append(projected)

            if num_point_clouds_valid is not None:
                projected_features = [
                    sample_feats[: int(num_point_clouds_valid[idx])]
                    if idx < len(num_point_clouds_valid)
                    else sample_feats
                    for idx, sample_feats in enumerate(projected_features)
                ]
            
            _log_feat_stats("point_proj_out", projected_features)

            # Apply relation module based on mode
            mode = (self.relation_mode or "object").lower()
            use_adaln = bool(getattr(self.relation_module, "use_adaln", False))
            
            # ★★★ DEBUG: AdaLN状態を確認 ★★★
            if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
                print(f"\n{'='*60}")
                print(f"[ADALN_DEBUG] Forward pass debug:")
                print(f"  use_adaln (from relation_module): {use_adaln}")
                print(f"  self.relation_use_adaln: {getattr(self, 'relation_use_adaln', 'NOT_SET')}")
                print(f"  input_ids is not None: {input_ids is not None}")
                print(f"  inputs_embeds is not None: {inputs_embeds is not None}")
                print(f"  Note: No projection needed (4096-dim → 4096-dim)")
                print(f"{'='*60}\n")
            
            if use_adaln and input_ids is not None and inputs_embeds is not None:
                # Prefill step: compute condition vectors
                cond_list = [
                    self._extract_condition_vector(ids, embeds, point_backbone_config)
                    for ids, embeds in zip(input_ids, inputs_embeds)
                ]
                if getattr(self, "fix_llm", False):
                    cond_list = [None if cond is None else cond.detach() for cond in cond_list]
                
                # CRITICAL: Cache condition vectors for KV-cache steps
                # During generation, inputs_embeds is None after the first token, but we still need cond
                self._cached_cond_list = [c.detach() if c is not None else None for c in cond_list]
                
                # ★★★ DEBUG: cond_list内容を確認 ★★★
                if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
                    none_count = sum(1 for c in cond_list if c is None)
                    print(f"[ADALN_DEBUG] cond_list created via _extract_condition_vector (PREFILL):")
                    print(f"  Total items: {len(cond_list)}")
                    print(f"  None items: {none_count}")
                    if none_count < len(cond_list):
                        sample_cond = next((c for c in cond_list if c is not None), None)
                        if sample_cond is not None:
                            print(f"  Sample cond shape: {sample_cond.shape} (4096-dim, same as relation features)")
                            print(f"  Sample cond dtype: {sample_cond.dtype}")
                            # 全てのcondの統計を計算
                            valid_conds = [c for c in cond_list if c is not None]
                            if valid_conds:
                                all_conds = torch.cat(valid_conds, dim=0)
                                print(f"  All conds mean: {all_conds.mean().item():.6f}")
                                print(f"  All conds std: {all_conds.std().item():.6f}")
                    print(f"  ✓ Cached for KV-cache steps")
                    print()
            else:
                # KV-cache step or AdaLN disabled: reuse cached conditions if available
                if use_adaln and hasattr(self, "_cached_cond_list"):
                    cond_list = self._cached_cond_list
                    if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
                        print(f"[ADALN_DEBUG] Reusing cached cond_list (KV-CACHE step)")
                        print(f"  Total items: {len(cond_list)}")
                        print(f"  None items: {sum(1 for c in cond_list if c is None)}")
                        print()
                else:
                    cond_list = [None] * len(projected_features)
                    
                    # ★★★ DEBUG: cond_list=None時の理由を出力 ★★★
                    if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
                        print(f"[ADALN_DEBUG] cond_list set to [None] * {len(projected_features)}")
                        print(f"  Reason: use_adaln={use_adaln}, input_ids is not None={input_ids is not None}, inputs_embeds is not None={inputs_embeds is not None}")
                        print(f"  has_cache={hasattr(self, '_cached_cond_list')}")
                        print()
            
            if self.relation_module is not None:
                if mode == "object":
                    projected_features = [
                        self._apply_relation_object(sample_feats, cond_list[idx])
                        for idx, sample_feats in enumerate(projected_features)
                    ]
                elif mode == "patch":
                    projected_features = [
                        self._apply_relation_patch(sample_feats, cond_list[idx])
                        for idx, sample_feats in enumerate(projected_features)
                    ]
                elif mode == "micro":
                    # A: micro-token relation with patch-wise delta (LLM receives original patches)
                    projected_features = [
                        self._apply_relation_micro(sample_feats, cond_list[idx])
                        for idx, sample_feats in enumerate(projected_features)
                    ]
                elif mode == "fast_patch":
                    # B: compress to micro tokens, apply relation, use micro tokens for LLM
                    micro_features = []
                    for sample_feats in projected_features:
                        micro_features.append([token_fps_pool(feat, M=self._micro_M, metric="cos") for feat in sample_feats])

                    updated_micro = []
                    for idx, sample_micro in enumerate(micro_features):
                        if len(sample_micro) == 0:
                            updated_micro.append(sample_micro)
                            continue
                        cat = torch.cat(sample_micro, dim=0).unsqueeze(0)  # (1, sum(M*obj_i), D)
                        cond_tensor = cond_list[idx]
                        if self.relation_use_adaln and cond_tensor is not None:
                            cond_tensor = cond_tensor.to(cat.device)
                        else:
                            cond_tensor = None
                        _log_feat_stats("relation_in(fast_patch)", [cat.squeeze(0)])
                        cat_upd = self.relation_module(cat, cond=cond_tensor).squeeze(0)
                        _log_feat_stats("relation_out(fast_patch)", [cat_upd])

                        ptr = 0
                        cur_upd = []
                        for P in sample_micro:
                            cur_upd.append(cat_upd[ptr:ptr + P.size(0)])
                            ptr += P.size(0)
                        updated_micro.append(cur_upd)
                    projected_features = updated_micro
                else:
                    # Default: object mode
                    projected_features = [
                        self._apply_relation_object(sample_feats, cond_list[idx])
                        for idx, sample_feats in enumerate(projected_features)
                    ]
            else:
                # No relation module: only apply compression for fast_patch mode
                if mode == "fast_patch":
                    micro_features = []
                    for sample_feats in projected_features:
                        micro_features.append([token_fps_pool(feat, M=self._micro_M, metric="cos") for feat in sample_feats])
                    projected_features = micro_features

            new_input_embeds = []
            for batch_idx, (cur_input_ids, cur_input_embeds) in enumerate(zip(input_ids, inputs_embeds)):
                if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() == 0 and (
                    point_backbone_config.get("mm_use_point_start_end", True)
                    and (cur_input_ids == point_backbone_config.get("point_start_token", -1)).sum() == 0
                ):
                    new_input_embeds.append(cur_input_embeds)
                    continue

                batch_point_features = projected_features[batch_idx]

                if point_backbone_config["mm_use_point_start_end"]:
                    starts = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    ends = torch.where(cur_input_ids == point_backbone_config["point_end_token"])[0]

                    if len(starts) != len(ends):
                        raise ValueError("Mismatch between <point_start> and <point_end> tokens.")
                    if len(starts) > len(batch_point_features):
                        raise ValueError(
                            f"Found {len(starts)} point regions but only {len(batch_point_features)} point clouds."
                        )

                    for idx in reversed(range(len(starts))):
                        start_idx, end_idx = starts[idx], ends[idx]
                        feature = batch_point_features[idx].to(cur_input_embeds.device)
                        
                        # Check length consistency before replacement
                        region_ids = cur_input_ids[start_idx + 1 : end_idx]
                        num_text_patches = int((region_ids == point_backbone_config["point_patch_token"]).sum().item())
                        if feature.shape[0] != num_text_patches:
                            raise RuntimeError(
                                f"[CVPR] Patch-token length mismatch: feature_len={feature.shape[0]} vs text_patch_tokens={num_text_patches}. "
                                "Check dataset replacement length and projection output."
                            )
                        
                        # Middle: <point_start> + feature + <point_end> (trainable)
                        mid = torch.cat(
                            [
                                cur_input_embeds[start_idx : start_idx + 1],
                                feature,
                                cur_input_embeds[end_idx : end_idx + 1],
                            ],
                            dim=0,
                        )
                        left = cur_input_embeds[:start_idx]
                        right = cur_input_embeds[end_idx + 1 :]
                        
                        # Match original PointLLM: detach left/right only when orig_embeds_params exists
                        if orig_embeds_params is not None:
                            left = left.detach()
                            right = right.detach()
                        
                        cur_input_embeds = torch.cat([left, mid, right], dim=0)
                else:
                    patch_positions = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    if not batch_point_features:
                        raise ValueError("No point features available for replacement.")
                    feature = batch_point_features[0].to(cur_input_embeds.device)
                    num_patches = feature.shape[0]
                    if len(patch_positions) != num_patches:
                        raise ValueError("Number of <point_patch> tokens does not match point patches.")
                    start_idx = patch_positions[0]
                    
                    # Match original PointLLM: detach left/right only when orig_embeds_params exists
                    if orig_embeds_params is not None:
                        cur_input_embeds = torch.cat(
                            [
                                cur_input_embeds[:start_idx].detach(),
                                feature,
                                cur_input_embeds[start_idx + num_patches :].detach(),
                            ],
                            dim=0,
                        )
                    else:
                        cur_input_embeds = torch.cat(
                            [
                                cur_input_embeds[:start_idx],
                                feature,
                                cur_input_embeds[start_idx + num_patches :],
                            ],
                            dim=0,
                        )

                new_input_embeds.append(cur_input_embeds)

            if len(new_input_embeds) > 1:
                lengths = [emb.shape[0] for emb in new_input_embeds]
                if not all(length == lengths[0] for length in lengths):
                    raise RuntimeError(f"Sequence length mismatch after point replacement: {lengths}")

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # For broad compatibility: accept cache_position but do not pass it to base.
        return super(PointLLMLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PointLLMCVPRLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMCVPRConfig

    def _init_weights(self, module):
        super()._init_weights(module)
        if getattr(module, "_pit_zero_output", False):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, config: LlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMCVPRLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if self.model.relation_module is not None:
            # post_init() initializes every Linear recursively. Restore PIT's
            # identity-preserving residual initialization for direct model
            # construction; from_pretrained() overwrites it with checkpoint data.
            self.model.relation_module.reset_parameters_for_training()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        num_point_clouds_valid: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds,
            num_point_clouds_valid=num_point_clouds_valid,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # デバッグ用: prepare_inputs_for_generation呼び出しを追跡
        if os.getenv("POINTLLM_DEBUG_FORWARD", "0") == "1":
            pc_in_kwargs = "point_clouds" in kwargs
            pc_value = kwargs.get("point_clouds", None)
            pc_status = "Present" if pc_value is not None else "None"
            print(f"[PREP] point_clouds in kwargs={pc_in_kwargs}, value={pc_status}", flush=True)
        
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
                "cache_position": kwargs.get("cache_position", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # Replicate original PointLLM logic here (do not delegate to inner model)
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = getattr(config, "mm_use_point_start_end", True)

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
        self._record_conversation_token_ids(tokenizer)

    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True, data_args=None):
        # Replicate original PointLLM logic here (do not delegate to inner model)
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = getattr(config, "mm_use_point_start_end", True)

        # 1) collect tokens to add
        tokens_to_add = []
        tokens_to_add.append(config.DEFAULT_POINT_PATCH_TOKEN)
        if data_args and getattr(data_args, 'point_identifiers', None):
            tokens_to_add.extend(data_args.point_identifiers)
        if mm_use_point_start_end:
            tokens_to_add.append(config.DEFAULT_POINT_START_TOKEN)
            tokens_to_add.append(config.DEFAULT_POINT_END_TOKEN)

        # 2) add tokens and resize embeddings
        unique_tokens = list(dict.fromkeys(tokens_to_add))
        num_new = tokenizer.add_tokens(unique_tokens, special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        # 3) save token ids
        point_backbone_config['default_point_patch_token'] = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([config.DEFAULT_POINT_PATCH_TOKEN])[0]
        if mm_use_point_start_end:
            point_backbone_config['default_point_start_token'] = config.DEFAULT_POINT_START_TOKEN
            point_backbone_config['default_point_end_token'] = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([config.DEFAULT_POINT_START_TOKEN])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([config.DEFAULT_POINT_END_TOKEN])[0]

        # 4) initialize new embeddings
        if num_new > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new].mean(dim=0, keepdim=True)

            input_embeddings[-num_new:] = input_embeddings_avg
            output_embeddings[-num_new:] = output_embeddings_avg

            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            if fix_llm:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                print(f"[CVPR] Fixed output embeddings; {num_new} new input tokens trainable.")
            else:
                self.get_model().orig_embeds_params = None
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True
                print("[CVPR] Input/Output embeddings trainable.")
        self._record_conversation_token_ids(tokenizer)

    def _record_conversation_token_ids(self, tokenizer):
        """Record multiple tokenization variants for robust conversation boundary detection.
        
        Vicuna v1.1 uses sep=" " so the actual format is " ASSISTANT:" (with leading space).
        However, we record multiple candidates to handle tokenizer differences robustly.
        """
        point_backbone_config = getattr(self.get_model(), "point_backbone_config", None)
        if tokenizer is None or point_backbone_config is None:
            return
        
        # Try multiple patterns (with/without spaces, with/without colons) for robustness
        assistant_candidates = [" ASSISTANT:", "ASSISTANT:", " ASSISTANT", "ASSISTANT"]
        user_candidates = [" USER:", "USER:", " USER", "USER"]
        
        def _encode_all_variants(candidates):
            """Encode all variants and return as list, sorted by length (longer = more specific)"""
            variants = []
            for candidate in candidates:
                try:
                    tokens = tokenizer.encode(candidate, add_special_tokens=False)
                    if tokens:
                        variants.append([int(t) for t in tokens])
                except Exception:
                    pass
            # Sort by length descending (prefer longer/more specific patterns)
            variants.sort(key=len, reverse=True)
            return variants
        
        assistant_variants = _encode_all_variants(assistant_candidates)
        user_variants = _encode_all_variants(user_candidates)
        
        # Store all variants for robust matching
        if assistant_variants:
            point_backbone_config["assistant_token_id_variants"] = assistant_variants
            # Keep first (longest) as representative for backward compatibility
            point_backbone_config["assistant_token_ids"] = assistant_variants[0]
        if user_variants:
            point_backbone_config["user_token_id_variants"] = user_variants
            point_backbone_config["user_token_ids"] = user_variants[0]


# Register with AutoConfig / AutoModel so the trainer can load via config
AutoConfig.register(PointLLMCVPRConfig.model_type, PointLLMCVPRConfig)
AutoModelForCausalLM.register(PointLLMCVPRConfig, PointLLMCVPRLlamaForCausalLM)
