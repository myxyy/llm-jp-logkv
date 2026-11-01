"""Offline projection/RoPE equivalence and hierarchical streaming checks."""
from llm_jp_logkv.settings import load_settings
load_settings()

import math

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from llm_jp_logkv.llama_attention import LlamaAttentionAdapter


def setup(dtype=torch.float64):
    torch.manual_seed(17)
    config = LlamaConfig(hidden_size=32, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'eager'
    source = LlamaAttention(config, 0).to(dtype).eval()
    rotary = LlamaRotaryEmbedding(config)
    x = torch.randn(2, 19, 32, dtype=dtype)
    return source, rotary, x


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
def test_dense_matches_llama_output_cache_and_gradients(dtype):
    source, rotary, x = setup(dtype)
    adapter = LlamaAttentionAdapter(source)
    assert adapter.q_proj is source.q_proj
    x.requires_grad_()
    positions = torch.arange(x.shape[1])[None]
    rope = rotary(x, positions)
    mask = torch.full((19, 19), torch.finfo(dtype).min, dtype=dtype).triu(1)[None, None]
    expected, _ = source(x, position_embeddings=rope, attention_mask=mask)
    actual, state = adapter(x, rope)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(grad, reference_grad, atol=tol, rtol=tol)
    assert state.payload[0].shape == (2, 2, 19, 8)
    pieces, state = [], None
    for start, end in [(0, 3), (3, 17), (17, 18), (18, 19)]:
        part = x[:, start:end]
        out, state = adapter(part, rotary(part, positions[:, start:end]), state)
        pieces.append(out)
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(state.payload[0], adapter(x, rope)[1].payload[0])


@pytest.mark.parametrize('chunk_size', [2, 4])
@pytest.mark.parametrize('no_level_bias', [False, True])
def test_logkv_streaming_gradients_and_state(chunk_size, no_level_bias):
    source, rotary, x = setup()
    adapter = LlamaAttentionAdapter(source, mode='logkv', chunk_size=chunk_size,
                                    no_level_bias=no_level_bias)
    x.requires_grad_()
    pos = torch.arange(19)[None]
    full, _ = adapter(x, rotary(x, pos))
    state, pieces = None, []
    for start, end in [(0, 1), (1, 7), (7, 18), (18, 19)]:
        part = x[:, start:end]
        out, state = adapter(part, rotary(part, pos[:, start:end]), state)
        pieces.append(out)
    stream = torch.cat(pieces, 1)
    torch.testing.assert_close(stream, full, atol=1e-12, rtol=1e-12)
    g1 = torch.autograd.grad(full.square().sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(stream.square().sum(), x)[0]
    torch.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-12)
    levels, offset = state.payload
    assert offset == state.offset == 19
    for level in levels:
        for tensor in level:
            assert tensor.shape[0] == 2 * 4  # Q heads, not shared KV heads.
            assert tensor.shape[1] < chunk_size
    assert set(adapter.state_dict()) == set(source.state_dict())


def test_logkv_independent_fp64_reference():
    # Deliberately sequential oracle: read all retained slots, then carry a
    # completed chunk upward. Does not call any LogKV implementation helper.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    source, rotary, x = setup()
    adapter = LlamaAttentionAdapter(source, mode='logkv', chunk_size=4)
    shape = (2, 19, -1, 8)
    q, k, v = [p(x).view(shape).transpose(1, 2)
               for p in (source.q_proj, source.k_proj, source.v_proj)]
    q, k = apply_rotary_pos_emb(q, k, *rotary(x, torch.arange(19)[None]))
    k, v = repeat_kv(k, 2), repeat_kv(v, 2)
    levels, outputs = [], []
    for t in range(19):
        keys, values, biases = [k[:, :, t]], [v[:, :, t]], [0.]
        for depth, slots in enumerate(levels):
            for _, sk, sv in slots:
                keys.append(sk)
                values.append(sv)
                biases.append(-depth * math.log(4))
        keys, values = torch.stack(keys, -2), torch.stack(values, -2)
        logits = (q[:, :, t, None] * keys).sum(-1) / 8 ** .5
        logits = logits + x.new_tensor(biases)
        outputs.append((logits.softmax(-1)[..., None] * values).sum(-2))
        item, depth = (q[:, :, t], k[:, :, t], v[:, :, t]), 0
        while True:
            if depth == len(levels):
                levels.append([])
            levels[depth].append(item)
            if len(levels[depth]) < 4:
                break
            sq, sk, sv = [torch.stack([slot[i] for slot in levels[depth]], -2)
                          for i in range(3)]
            weights = ((sq[..., -1:, :] * sk).sum(-1) / 8 ** .5).softmax(-1)
            item = (sq[..., -1, :], (weights[..., None] * sk).sum(-2),
                    (weights[..., None] * sv).sum(-2))
            levels[depth] = []
            depth += 1
    expected = source.o_proj(torch.stack(outputs, 2).transpose(1, 2).reshape(2, 19, 32))
    actual, _ = adapter(x, rotary(x, torch.arange(19)[None]))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
