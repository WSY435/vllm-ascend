#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Ascend adaptation patch for MiniCPM3.

MiniCPM3 adopts a DeepSeek-V2 style Multi-head Latent Attention (MLA) whose
``qk_head_dim`` equals ``qk_nope_head_dim + qk_rope_head_dim`` (64 + 32 = 96 for
MiniCPM3-4B). The upstream vLLM implementation computes attention with the naive
path that pads ``v`` up to ``qk_head_dim`` and feeds ``q/k/v`` of head size 96 to
the ``Attention`` layer.

On Ascend, ``npu_fused_infer_attention_score`` (and paged attention) with the TND
layout only supports head sizes in {64, 128, 192} (or the special 192/128 case),
so head size 96 raises ``aclnnFusedInferAttentionScoreV3`` tiling errors.

This patch overrides ``MiniCPM3Attention`` so that ``q``, ``k`` and ``v`` are
zero-padded to the nearest supported head size (128 for MiniCPM3-4B) before being
passed to the ``Attention`` layer. Because the padded dims are filled with zeros,
they contribute nothing to the ``q @ k^T`` dot product nor to the
``softmax(scores) @ v`` output, so the numerical result is identical to the
upstream implementation while satisfying the Ascend kernel shape constraint.

Related issue: https://github.com/vllm-project/vllm-ascend/issues/10676
"""

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models import minicpm3

# Head sizes supported by Ascend fused infer attention with TND layout.
_ASCEND_SUPPORTED_HEAD_SIZES = (64, 128, 192)


def _ceil_to_supported_head_size(head_size: int) -> int:
    """Return the smallest Ascend-supported head size >= ``head_size``.

    If ``head_size`` is already supported (or larger than every supported
    value, in which case there is nothing we can pad to), it is returned
    unchanged.
    """
    for supported in _ASCEND_SUPPORTED_HEAD_SIZES:
        if supported >= head_size:
            return supported
    return head_size


class AscendMiniCPM3Attention(minicpm3.MiniCPM3Attention):
    """Ascend-friendly MiniCPM3 MLA attention.

    Identical projections to the upstream implementation, but the ``Attention``
    layer is created with a head size that Ascend supports, and ``q/k/v`` are
    zero-padded to that size in ``forward``.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Pad qk_head_dim up to the nearest Ascend-supported head size.
        self.padded_head_dim = _ceil_to_supported_head_size(self.qk_head_dim)
        if self.padded_head_dim != self.qk_head_dim:
            # Rebuild the Attention layer with the padded head size so that the
            # KV cache and the attention kernel agree on head_size. The scaling
            # factor is kept as qk_head_dim**-0.5: the padded dims are zeros and
            # do not contribute to the dot product, so the effective dimension is
            # still qk_head_dim.
            #
            # The parent __init__ already registered an Attention under
            # ``{prefix}.attn`` in the compilation forward context; drop that
            # entry first so the recreated layer does not trip the
            # "Duplicate layer name" check.
            attn_prefix = f"{prefix}.attn"
            forward_ctx = get_current_vllm_config(
            ).compilation_config.static_forward_context
            forward_ctx.pop(attn_prefix, None)
            self.attn = Attention(
                self.num_local_heads,
                self.padded_head_dim,
                self.scaling,
                num_kv_heads=self.num_local_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=attn_prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_a_proj(hidden_states)
        q = self.q_a_layernorm(q)
        q, _ = self.q_b_proj(q)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv, _ = self.kv_b_proj(kv_a)
        kv = kv.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(
            positions,
            q_pe.reshape(-1, self.num_local_heads * self.qk_rope_head_dim),
            k_pe.reshape(-1, self.qk_rope_head_dim),
        )
        q_pe = q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

        # Assemble q and k with the decoupled RoPE (nope + rope). Use cat instead
        # of in-place index assignment to stay friendly to graph capture.
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1
        )

        # Zero-pad q/k/v to the Ascend-supported head size. The padded dims are
        # zeros, so the attention scores and output are numerically unchanged.
        padded_head_dim = self.padded_head_dim
        if padded_head_dim != self.qk_head_dim:
            pad_qk = padded_head_dim - self.qk_head_dim
            q = F.pad(q, [0, pad_qk])
            k = F.pad(k, [0, pad_qk])
        v = F.pad(v, [0, padded_head_dim - self.v_head_dim])

        q = q.reshape(-1, self.num_local_heads * padded_head_dim)
        k = k.reshape(-1, self.num_local_heads * padded_head_dim)
        v = v.reshape(-1, self.num_local_heads * padded_head_dim)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(
            -1, self.num_local_heads, padded_head_dim
        )[..., : self.v_head_dim].reshape(
            -1, self.num_local_heads * self.v_head_dim
        )

        output, _ = self.o_proj(attn_output)
        return output


# Monkey-patch the upstream class so that MiniCPM3DecoderLayer._init_attn_block
# (which references MiniCPM3Attention as a module global) picks up the
# Ascend-friendly implementation when the model is constructed on the worker.
minicpm3.MiniCPM3Attention = AscendMiniCPM3Attention
