# SPDX-License-Identifier: Apache-2.0
"""Performance patches for the external `cosyvoice` package's Flow (DiT) estimator.

sglang-omni does not vendor the `cosyvoice` package (see
`_load_cosyvoice3_flow_hift` in stages.py), so these fixes are applied as
runtime monkey-patches after import rather than as edits to that package.

Profiling of `flow.inference()` (see the PR #1715 / issue #1652 discussion)
found that RoPE sin/cos and the attention padding-mask negation are
recomputed at every (layer, ODE-step) pair -- 22 layers x 10 Euler steps =
220 times per call -- even though both only depend on `token_len`, which is
constant across the whole call. This module caches both, scoped to a single
`flow.inference()` call via a thread-local cache that is reset at entry/exit,
so it never persists (and can't go stale) across separate requests.

Note: the `neg` op inside RoPE's `rotate_half` operates on the actual
query/key tensors, which differ every layer -- that is real, necessary work
and is intentionally left untouched. Only `freqs.cos()`/`freqs.sin()` (which
depend solely on the shared `rope` tuple) and the mask negation (which
depends solely on the shared `attn_mask` tensor) are cached here.

Caveat: this assumes `flow.inference()` is called with a single, unpadded
sequence (true for `_CosyVoice3Vocoder._token2wav` today, which processes one
item at a time). If Flow ever gains true batched/padded inference, this
per-call cache is still correct (it never looks past one `inference()` call
or assumes anything about batch content), so no change is needed there
either way.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn.functional as F

_tls = threading.local()


def _get_cache() -> dict[Any, Any]:
    cache = getattr(_tls, "cache", None)
    if cache is None:
        cache = {}
        _tls.cache = cache
    return cache


def _reset_cache() -> None:
    _tls.cache = {}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def _cached_apply_rotary_pos_emb(
    t: torch.Tensor, freqs: torch.Tensor, scale: float | torch.Tensor = 1
) -> torch.Tensor:
    """Same math as x_transformers.x_transformers.apply_rotary_pos_emb, but
    caches freqs.cos()/freqs.sin() instead of recomputing them every call."""
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    cache = _get_cache()
    key = ("rope_trig", id(freqs), seq_len)
    trig = cache.get(key)
    if trig is None:
        freqs_sliced = freqs[:, -seq_len:, :]
        trig = (freqs_sliced.cos(), freqs_sliced.sin())
        cache[key] = trig
    cos, sin = trig

    scale_val = scale[:, -seq_len:, :] if torch.is_tensor(scale) else scale

    if t.ndim == 4 and cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        if torch.is_tensor(scale_val):
            scale_val = scale_val.unsqueeze(1)

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos * scale_val) + (_rotate_half(t) * sin * scale_val)
    out = torch.cat((t, t_unrotated), dim=-1)
    return out.type(orig_dtype)


def _cached_negated_mask(mask: torch.Tensor) -> torch.Tensor:
    cache = _get_cache()
    key = ("neg_mask", id(mask))
    neg_mask = cache.get(key)
    if neg_mask is None:
        if mask.dim() == 2:
            m = mask.unsqueeze(-1)
        else:
            m = mask[:, 0, -1].unsqueeze(-1)
        neg_mask = ~m
        cache[key] = neg_mask
    return neg_mask


def _patched_attn_processor_call(self, attn, x, mask=None, rope=None):
    batch_size = x.shape[0]

    query = attn.to_q(x)
    key = attn.to_k(x)
    value = attn.to_v(x)

    if rope is not None:
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (
            (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        )
        query = _cached_apply_rotary_pos_emb(query, freqs, q_xpos_scale)
        key = _cached_apply_rotary_pos_emb(key, freqs, k_xpos_scale)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if mask is not None:
        attn_mask = mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(
                batch_size, attn.heads, query.shape[-2], key.shape[-2]
            )
    else:
        attn_mask = None

    x = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )
    x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    x = x.to(query.dtype)

    x = attn.to_out[0](x)
    x = attn.to_out[1](x)

    if mask is not None:
        x = x.masked_fill(_cached_negated_mask(mask), 0.0)

    return x


def _make_patched_dit_forward(add_optional_chunk_mask):
    def _patched_dit_forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        cache = _get_cache()
        cache_key = ("rope_mask", id(self), seq_len, streaming)
        entry = cache.get(cache_key)
        if entry is None:
            rope = self.rotary_embed.forward_from_seq_len(seq_len)
            if streaming is True:
                attn_mask = add_optional_chunk_mask(
                    x, mask.bool(), False, False, 0, self.static_chunk_size, -1
                ).unsqueeze(dim=1)
            else:
                attn_mask = (
                    add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1)
                    .repeat(1, x.size(1), 1)
                    .unsqueeze(dim=1)
                )
            attn_mask = attn_mask.bool()
            entry = (rope, attn_mask)
            cache[cache_key] = entry
        rope, attn_mask = entry

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output

    return _patched_dit_forward


@torch.compile(dynamic=True)
def _fused_modulate(
    normed: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """`normed * (1 + scale) + shift`, fused into one Inductor kernel instead
    of separate aten::mul + aten::add launches."""
    return normed * (1 + scale) + shift


@torch.compile(dynamic=True)
def _fused_gated_residual(
    x: torch.Tensor, gate: torch.Tensor, sublayer_out: torch.Tensor
) -> torch.Tensor:
    """`x + gate * sublayer_out`, fused into one Inductor kernel."""
    return x + gate * sublayer_out


def _patched_ada_ln_zero_forward(self, x, emb=None):
    emb = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
        emb, 6, dim=1
    )
    x = _fused_modulate(self.norm(x), scale_msa[:, None], shift_msa[:, None])
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _patched_ada_ln_zero_final_forward(self, x, emb):
    emb = self.linear(self.silu(emb))
    scale, shift = torch.chunk(emb, 2, dim=1)
    return _fused_modulate(self.norm(x), scale[:, None, :], shift[:, None, :])


def _patched_dit_block_forward(self, x, t, mask=None, rope=None):
    norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

    attn_output = self.attn(x=norm, mask=mask, rope=rope)
    x = _fused_gated_residual(x, gate_msa.unsqueeze(1), attn_output)

    ff_norm = _fused_modulate(self.ff_norm(x), scale_mlp[:, None], shift_mlp[:, None])
    ff_output = self.ff(ff_norm)
    x = _fused_gated_residual(x, gate_mlp.unsqueeze(1), ff_output)

    return x


_applied = False


def apply_dit_perf_patches(enable_adaln_fusion: bool | None = None) -> None:
    """Idempotently monkey-patch the external cosyvoice DiT estimator to
    stop recomputing per-call-constant RoPE/mask values on every layer and
    ODE step. Call once, after `cosyvoice` has been imported.

    `enable_adaln_fusion` gates the separate AdaLN-modulate/gated-residual
    torch.compile fusion (Item 2b) so it can be A/B'd in isolation from the
    RoPE/mask caching (Item 2a); defaults to the
    SGLANG_OMNI_COSYVOICE3_ADALN_FUSION env var (default: enabled) so
    profiling scripts run as separate processes can toggle it without
    editing this file.
    """
    global _applied
    if _applied:
        return

    if enable_adaln_fusion is None:
        import os

        enable_adaln_fusion = os.environ.get(
            "SGLANG_OMNI_COSYVOICE3_ADALN_FUSION", "1"
        ) not in ("0", "false", "False")

    from cosyvoice.flow.DiT import modules as dit_modules
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.utils.mask import add_optional_chunk_mask

    dit_modules.AttnProcessor.__call__ = _patched_attn_processor_call
    if enable_adaln_fusion:
        dit_modules.AdaLayerNormZero.forward = _patched_ada_ln_zero_forward
        dit_modules.AdaLayerNormZero_Final.forward = _patched_ada_ln_zero_final_forward
        dit_modules.DiTBlock.forward = _patched_dit_block_forward
    DiT.forward = _make_patched_dit_forward(add_optional_chunk_mask)

    orig_inference = CausalMaskedDiffWithDiT.inference

    def _patched_inference(self, *args, **kwargs):
        _reset_cache()
        try:
            return orig_inference(self, *args, **kwargs)
        finally:
            _reset_cache()

    CausalMaskedDiffWithDiT.inference = _patched_inference

    _applied = True
