"""Experimental Llama projection adapter; not a Transformers Cache replacement.

Call load_settings before importing this module in standalone programs.
"""
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .logkv.attention import LogKV


@dataclass(frozen=True)
class AttentionState:
    mode: str
    offset: int
    payload: object


class LlamaAttentionAdapter(nn.Module):
    """Unpadded causal attention with explicit, caller-owned streaming state.

    Reuses the source projections by identity. RoPE is supplied by the original
    Llama rotary module using positions starting at state.offset. No conv/gate,
    normalization or extra positional transform is introduced.
    """

    def __init__(self, source, *, mode="dense", chunk_size=4, no_level_bias=False):
        super().__init__()
        if mode not in ("dense", "logkv"):
            raise ValueError("mode must be dense or logkv")
        if source.attention_dropout != 0:
            raise ValueError("Only zero attention dropout is supported")
        self.mode = mode
        self.head_dim = source.head_dim
        self.heads = source.config.num_attention_heads
        self.groups = source.num_key_value_groups
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(source, name))
        # Keep reference semantics intact. Projection layers are unused by the
        # projected-tensor entry point, so remove their parameters here only.
        self.core = None
        if mode == "logkv":
            self.core = LogKV(self.heads * self.head_dim, chunk_size,
                              num_heads=self.heads, self_slot=True,
                              no_level_bias=no_level_bias)
            for name in ("lq", "lk", "lv", "lo"):
                setattr(self.core, name, nn.Identity())

    def forward(self, hidden_states, position_embeddings, state=None):
        if hidden_states.ndim != 3 or hidden_states.shape[1] == 0:
            raise ValueError("Expected nonempty [batch, sequence, hidden] input")
        if state is not None and state.mode != self.mode:
            raise ValueError("State mode does not match adapter")
        batch, length, _ = hidden_states.shape
        offset = 0 if state is None else state.offset
        shape = (batch, length, -1, self.head_dim)
        q, k, v = [proj(hidden_states).view(shape).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if self.mode == "dense":
            if state is not None:
                old_k, old_v = state.payload
                k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
            payload = (k, v)  # Native GQA heads retained in dense cache.
            keys, values = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
            logits = q @ keys.transpose(-1, -2) * self.head_dim ** -0.5
            causal = torch.arange(k.shape[2], device=q.device)[None, :] > (
                offset + torch.arange(length, device=q.device)[:, None])
            logits = logits.masked_fill(causal, torch.finfo(q.dtype).min)
            # Match Llama eager's explicit FP32 softmax, including FP64 tests.
            out = logits.softmax(-1, dtype=torch.float32).to(q.dtype) @ values
        else:
            k, v = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
            def fold(t):
                return t.reshape(batch * self.heads, length, self.head_dim)
            out, payload = self.core._attend(
                fold(q), fold(k), fold(v), None if state is None else state.payload)
            out = out.view(batch, self.heads, length, self.head_dim)
        out = out.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        return self.o_proj(out), AttentionState(self.mode, offset + length, payload)
