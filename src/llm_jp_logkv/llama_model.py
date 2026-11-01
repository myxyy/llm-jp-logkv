"""Llama weights with explicit per-layer dense or LogKV inference state.

Standalone API: unpadded, contiguous sequences on a single device. It does not
implement Transformers Cache or model.generate. Load settings before import.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .llama_attention import AttentionState, LlamaAttentionAdapter


@dataclass(frozen=True)
class ModelState:
    layers: tuple[AttentionState, ...]
    offset: int

    @property
    def tensor_bytes(self) -> int:
        """Logical state tensor bytes (not allocator peak or model memory)."""
        total = 0
        for state in self.layers:
            tensors = (state.payload if state.mode == 'dense'
                       else [t for level in state.payload[0] for t in level])
            total += sum(t.numel() * t.element_size() for t in tensors)
        return total


class LlamaLogKVForCausalLM(nn.Module):
    """Reuse embeddings, RoPE, norms, FFNs, LM head and all attention weights.

    All layers use the selected mode. The source model is not modified and
    weights are shared, so source and wrapper must not be trained independently.
    No distillation, quantization, or new learned parameters are involved.
    """

    def __init__(self, source, *, mode='logkv', chunk_size=4, no_level_bias=False):
        super().__init__()
        self.config = source.config
        self.embed_tokens = source.model.embed_tokens
        self.rotary_emb = source.model.rotary_emb
        self.norm = source.model.norm
        self.lm_head = source.lm_head
        self.layers = source.model.layers
        self.attentions = nn.ModuleList([
            LlamaAttentionAdapter(layer.self_attn, mode=mode, chunk_size=chunk_size,
                                  no_level_bias=no_level_bias)
            for layer in self.layers
        ])
        self.mode = mode
        self.chunk_size = chunk_size
        self.no_level_bias = no_level_bias
        self.eval()

    def forward(self, input_ids, state=None, *, logits_to_keep=0):
        """Return logits and new state, leaving the supplied state untouched.

        logits_to_keep=0 returns all positions; a positive value limits LM-head
        computation and memory, especially for a large vocabulary during decode.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('Expected nonempty [batch, sequence] token IDs')
        if logits_to_keep < 0:
            raise ValueError('logits_to_keep must be nonnegative')
        if state is not None:
            if len(state.layers) != len(self.layers):
                raise ValueError('Wrong number of cached layers')
            if any(s.offset != state.offset for s in state.layers):
                raise ValueError('Inconsistent cache offsets')
        offset = 0 if state is None else state.offset
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(offset, offset + input_ids.shape[1], device=hidden.device)[None]
        rope = self.rotary_emb(hidden, positions)
        new_states = []
        for i, (layer, attention) in enumerate(zip(self.layers, self.attentions)):
            residual = hidden
            out, new_state = attention(layer.input_layernorm(hidden), rope,
                                       None if state is None else state.layers[i])
            hidden = residual + out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            new_states.append(new_state)
        hidden = self.norm(hidden)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = self.lm_head(hidden)
        return logits, ModelState(tuple(new_states), offset + input_ids.shape[1])

    @torch.inference_mode()
    def greedy_generate(self, input_ids, *, max_new_tokens=128, eos_token_id=None,
                        prefill_chunk_size=128):
        """Single unpadded prompt, deterministic greedy decode with EOS stop.

        Returns generated token IDs (including an encountered EOS), final state,
        and stop reason. Final state covers every returned generated token.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError('Generation requires one nonempty, unpadded prompt')
        if max_new_tokens < 1 or prefill_chunk_size < 1:
            raise ValueError('Token limits must be positive')
        eos = set([] if eos_token_id is None else
                  ([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id))
        state = None
        for part in input_ids.split(prefill_chunk_size, dim=1):
            logits, state = self(part, state, logits_to_keep=1)
        generated, reason = [], 'length'
        for _ in range(max_new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            # Consume final token as well so the returned state is resumable.
            logits, state = self(token, state, logits_to_keep=1)
            if token.item() in eos:
                reason = 'eos'
                break
        return torch.cat(generated, dim=1), state, reason
