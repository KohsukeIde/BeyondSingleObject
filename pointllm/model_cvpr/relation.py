import os
import numpy as np
import torch
from einops import rearrange
from torch import nn
from typing import List, Optional


def _prepare_attn_mask(mask: Optional[torch.Tensor], heads: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    if mask.dim() == 2:
        mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,T,T)

    if mask.size(1) == 1 and heads > 1:
        mask = mask.expand(-1, heads, -1, -1)

    return mask


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    attn_mask = _prepare_attn_mask(mask, q.size(1))
    if attn_mask is not None:
        attn_mask = attn_mask.to(device=q.device, dtype=torch.bool)

    try:
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            # This module uses True=masked; SDPA boolean masks use True=allowed.
            attn_mask=None if attn_mask is None else ~attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if torch.isnan(out).any():
            raise ValueError('sdpa produced NaNs')
        return out
    except (RuntimeError, ValueError) as e:
        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

        if attn_mask is not None:
            neg_inf = -torch.finfo(scaled_dot_prod.dtype).max
            scaled_dot_prod = scaled_dot_prod.masked_fill(attn_mask, neg_inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        out = self.W_0(out)
        
        return out


class AdaLNZero(nn.Module):
    """LayerNorm followed by conditioning-driven scale and shift (AdaLN-Zero)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * (1.0 + scale) + shift


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    cond が与えられた場合は AdaLN-Zero で質問依存のモジュレーションを追加。
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        # LayerNormのepsを大きくして数値安定性を向上（1e-5 -> 1e-3）
        self.norm_1 = nn.LayerNorm(dim, eps=1e-3)
        self.norm_2 = nn.LayerNorm(dim, eps=1e-3)
        self.adaln_msa = AdaLNZero(dim)
        self.adaln_ffn = AdaLNZero(dim)
        self._last_stats = None

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None,
                msa_scale: Optional[torch.Tensor] = None,
                msa_shift: Optional[torch.Tensor] = None,
                msa_gate: Optional[torch.Tensor] = None,
                ffn_scale: Optional[torch.Tensor] = None,
                ffn_shift: Optional[torch.Tensor] = None,
                ffn_gate: Optional[torch.Tensor] = None):
        # AdaLNが使用されない場合は元の処理を実行
        if msa_scale is None or msa_shift is None or msa_gate is None:
            if self.prenorm:
                # Pre-norm: norm → mhsa → dropout → residual
                normed = self.norm_1(x)
                mhsa_out = self.mhsa(normed, mask)
                dropped = self.drop(mhsa_out)
                y = dropped + x
                
                normed2 = self.norm_2(y)
                ffn_out = self.linear(normed2)
                out = ffn_out + y
            else:
                mhsa_out = self.mhsa(x, mask)
                dropped = self.drop(mhsa_out)
                residual1 = dropped + x
                y = self.norm_1(residual1)
                
                ffn_out = self.linear(y)
                residual2 = ffn_out + y
                out = self.norm_2(residual2)
            
            self._last_stats = None
            return out
        
        # AdaLNを使用する処理
        y = self.adaln_msa(x, msa_scale, msa_shift)
        attn_out = self.mhsa(y, mask)
        if msa_gate is None:
            msa_gate = 1.0
        added_attn = self.drop(attn_out * msa_gate)
        x = x + added_attn
        
        y_ffn = self.adaln_ffn(x, ffn_scale, ffn_shift)
        if ffn_gate is None:
            ffn_gate = 1.0
        ffn_out = self.linear(y_ffn)
        added_ffn = self.drop(ffn_out * ffn_gate)
        out = x + added_ffn

        if self.training:
            try:
                msa_stat = added_attn.detach().abs().mean().item()
                ffn_stat = added_ffn.detach().abs().mean().item()
            except Exception:
                msa_stat = ffn_stat = float("nan")
            self._last_stats = (msa_stat, ffn_stat)
        else:
            self._last_stats = None
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=1, heads=16, dim_head=None, dim_linear_layer=4096, dropout=0.1, prenorm=False, use_adaln=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_layer, dropout, prenorm=prenorm) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.block_list)
        self.use_adaln = bool(use_adaln)
        self.prenorm = bool(prenorm)
        if self.use_adaln:
            self.modulator = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, num_layers * 6 * dim)
            )
        else:
            self.modulator = None
        self._debug_gate_stats = False
        self._debug_gate_every = 1
        self._debug_gate_step = 0
        self.reset_parameters_for_training()

    def reset_parameters_for_training(self) -> None:
        """Initialize an identity residual block without killing its gradients."""
        self.apply(weight_init)

        if self.use_adaln:
            # AdaLN-Zero starts with closed residual gates. The transformer
            # weights become trainable as soon as the gates move away from zero.
            nn.init.zeros_(self.modulator[1].weight)
            nn.init.zeros_(self.modulator[1].bias)
            return

        # Keep QKV and the FFN input projection normally initialized. Zero only
        # each residual branch's output projection. This is exactly identity at
        # initialization while preserving a staged gradient path.
        for layer in self.layers:
            nn.init.zeros_(layer.mhsa.W_0.weight)
            if layer.mhsa.W_0.bias is not None:
                nn.init.zeros_(layer.mhsa.W_0.bias)

            ffn_output = layer.linear[3]
            nn.init.zeros_(ffn_output.weight)
            if ffn_output.bias is not None:
                nn.init.zeros_(ffn_output.bias)

    def forward(self, x, mask=None, cond: Optional[torch.Tensor] = None):
        # ★★★ DEBUG: TransformerEncoderでの受け取り確認 ★★★
        import os
        if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
            print(f"[ADALN_DEBUG] TransformerEncoder.forward():")
            print(f"  self.use_adaln: {self.use_adaln}")
            print(f"  cond is not None: {cond is not None}")
            if cond is not None:
                print(f"  cond shape: {cond.shape}")
                print(f"  cond dtype: {cond.dtype}")
            print()
        
        mod = None
        if self.use_adaln:
            if cond is None:
                # CRITICAL FIX: If cond is None (e.g., ASSISTANT: token not found during inference),
                # use zero-condition to trigger AdaLN-Zero identity behavior (scale=1, shift=0, gate=0).
                # This prevents falling back to random-initialized standard Transformer blocks.
                cond = torch.zeros(x.size(0), x.size(-1), dtype=x.dtype, device=x.device)
                if os.getenv("POINTLLM_DEBUG_ADALN", "0") == "1":
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("[AdaLN] Using zero-condition fallback (cond was None)")
            mod = self.modulator(cond)
            mod = mod.view(cond.size(0), len(self.layers), 6, x.size(-1))

        debug_enabled = bool(self._debug_gate_stats and mod is not None)
        if debug_enabled:
            try:
                every = max(1, int(self._debug_gate_every))
            except Exception:
                every = 1
            self._debug_gate_step += 1
            step_idx = self._debug_gate_step
            should_print = (step_idx % every) == 0
            x_input_snapshot = x.detach().clone()
            layer_stats = []
        else:
            should_print = False
            layer_stats = []
            x_input_snapshot = None

        for idx, layer in enumerate(self.layers):
            if mod is None:
                x = layer(x, mask)
            else:
                msa_scale = mod[:, idx, 0].unsqueeze(1)
                msa_shift = mod[:, idx, 1].unsqueeze(1)
                msa_gate = torch.tanh(mod[:, idx, 2]).unsqueeze(1)
                ffn_scale = mod[:, idx, 3].unsqueeze(1)
                ffn_shift = mod[:, idx, 4].unsqueeze(1)
                ffn_gate = torch.tanh(mod[:, idx, 5]).unsqueeze(1)
                x = layer(
                    x,
                    mask,
                    msa_scale=msa_scale,
                    msa_shift=msa_shift,
                    msa_gate=msa_gate,
                    ffn_scale=ffn_scale,
                    ffn_shift=ffn_shift,
                    ffn_gate=ffn_gate,
                )
                if debug_enabled:
                    last_stats = getattr(layer, "_last_stats", None)
                    if last_stats is not None:
                        layer_stats.append((idx, last_stats))

        if debug_enabled and should_print and layer_stats:
            try:
                msa_vals = [stat[1][0] for stat in layer_stats]
                ffn_vals = [stat[1][1] for stat in layer_stats]
                msa_avg = float(sum(msa_vals) / len(msa_vals))
                ffn_avg = float(sum(ffn_vals) / len(ffn_vals))
                msa_max = float(max(msa_vals))
                ffn_max = float(max(ffn_vals))
                total_delta = (x - x_input_snapshot).detach().abs().mean().item()
                print(
                    f"[RELATION][call={self._debug_gate_step}] "
                    f"msa|Δ| avg={msa_avg:.6f} max={msa_max:.6f} "
                    f"ffn|Δ| avg={ffn_avg:.6f} max={ffn_max:.6f} "
                    f"total|Δ|={total_delta:.6f} layers={len(layer_stats)}"
                )
                self._last_debug_metrics = {
                    "msa_delta_avg": msa_avg,
                    "msa_delta_max": msa_max,
                    "ffn_delta_avg": ffn_avg,
                    "ffn_delta_max": ffn_max,
                    "total_delta_avg": total_delta,
                    "layer_count": float(len(layer_stats)),
                }
            except Exception as exc:
                print(f"[RELATION] debug logging failed: {exc}")
                self._last_debug_metrics = None
        else:
            self._last_debug_metrics = None

        return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        # Xavier uniform初期化（元のコード）
        # nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(m.weight, mean=0.0, std=0.015)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # LayerNormの初期化（非常に重要！）
        nn.init.ones_(m.weight)  # gamma = 1
        nn.init.zeros_(m.bias)   # beta = 0


class SimpleRelationModule(nn.Module):
    """Chat-3D style Transformer encoder over token sets (copied behavior)."""

    def __init__(self, hidden_size: int, num_layers: int = 1, num_heads: int = 8, dropout: float = 0.1, prenorm: bool = False, use_adaln: bool = False):
        super().__init__()
        self.use_adaln = bool(use_adaln)
        if not self.use_adaln:
            prenorm = True  # ensure zero-init keeps the module at identity
        self.encoder = TransformerEncoder(
            dim=hidden_size,
            num_layers=num_layers,
            heads=num_heads,
            dim_linear_layer=hidden_size * 4,
            dropout=dropout,
            prenorm=prenorm,
            use_adaln=self.use_adaln,
        )
        debug_flag = os.environ.get("POINTLLM_RELATION_DEBUG", "").strip().lower()
        self.debug_gate_stats = self.use_adaln and debug_flag not in ("", "0", "false", "no")
        if self.debug_gate_stats:
            try:
                every = int(os.environ.get("POINTLLM_RELATION_DEBUG_EVERY", "1"))
            except ValueError:
                every = 1
            self.debug_gate_every = max(1, every)
            print(f"[RELATION] Gate debug logging enabled (every {self.debug_gate_every} calls).")
        else:
            self.debug_gate_every = 1
        self.encoder._debug_gate_stats = self.debug_gate_stats
        self.encoder._debug_gate_every = self.debug_gate_every

    def reset_parameters_for_training(self) -> None:
        self.encoder.reset_parameters_for_training()

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Accept (B, T) mask -> build (B,1,T,T) boolean mask where True=masked
        mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                B, T = key_padding_mask.shape
                mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)
            else:
                mask = key_padding_mask
        if not self.use_adaln:
            cond = None
        elif cond is not None:
            cond = cond.to(tokens.dtype).to(tokens.device)
        return self.encoder(tokens, mask=mask, cond=cond)
