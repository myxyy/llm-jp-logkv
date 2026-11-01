"""Full-model equivalence, causal streaming, generation and weight reuse."""
from llm_jp_logkv.settings import load_settings
load_settings()

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_jp_logkv.llama_model import LlamaLogKVForCausalLM


def tiny(dtype=torch.float32):
    torch.manual_seed(29)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=56,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         eos_token_id=[2, 13], pad_token_id=0)
    config._attn_implementation = 'eager'
    source = LlamaForCausalLM(config).to(dtype).eval()
    ids = torch.tensor([[1, 9, 7, 8, 12, 19, 20, 21, 5]])
    return source, ids


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_full_dense_matches_original_llama_and_generation(dtype):
    source, ids = tiny(dtype)
    model = LlamaLogKVForCausalLM(source, mode='dense')
    expected = source(ids, use_cache=False).logits
    actual, state = model(ids)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    out, state = model(ids[:, :3])
    tail, state = model(ids[:, 3:], state)
    torch.testing.assert_close(torch.cat((out, tail), 1), expected, atol=tol, rtol=tol)
    baseline = source.generate(ids, do_sample=False, max_new_tokens=5,
                               eos_token_id=None, pad_token_id=0)
    generated, state, _ = model.greedy_generate(ids, max_new_tokens=5, prefill_chunk_size=3)
    torch.testing.assert_close(generated, baseline[:, ids.shape[1]:])
    assert state.offset == ids.shape[1] + generated.shape[1]


@pytest.mark.parametrize('no_level_bias', [False, True])
@torch.no_grad()
def test_all_logkv_layers_streaming_causality_and_weight_identity(no_level_bias):
    source, ids = tiny(torch.float64)
    model = LlamaLogKVForCausalLM(source, no_level_bias=no_level_bias)
    assert model.embed_tokens is source.model.embed_tokens
    assert model.lm_head is source.lm_head
    for i, layer in enumerate(source.model.layers):
        assert model.layers[i].mlp is layer.mlp
        assert model.layers[i].input_layernorm is layer.input_layernorm
        assert model.attentions[i].q_proj is layer.self_attn.q_proj
    # No new trainable parameters; all original parameters remain in use.
    assert {id(p) for p in model.parameters()} == {id(p) for p in source.parameters()}
    full, final = model(ids)
    prefix, old = model(ids[:, :3])
    snapshot = [[t.clone() for level in s.payload[0] for t in level] for s in old.layers]
    tail, state = model(ids[:, 3:], old)
    torch.testing.assert_close(torch.cat((prefix, tail), 1), full, atol=1e-12, rtol=1e-12)
    for saved, s in zip(snapshot, old.layers):
        for before, after in zip(saved, [t for level in s.payload[0] for t in level]):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
    changed = ids.clone()
    changed[:, -1] = 6
    other, _ = model(changed)
    torch.testing.assert_close(other[:, :-1], full[:, :-1], atol=0, rtol=0)
    assert final.tensor_bytes == state.tensor_bytes > 0
    generated, final, reason = model.greedy_generate(ids, max_new_tokens=4, prefill_chunk_size=2)
    all_ids = torch.cat((ids, generated), 1)
    next_logits, _ = model(all_ids, logits_to_keep=1)
    resumed, _ = model(all_ids[:, -1:], model(all_ids[:, :-1])[1], logits_to_keep=1)
    torch.testing.assert_close(resumed, next_logits, atol=1e-12, rtol=1e-12)
    assert final.offset == all_ids.shape[1]


@torch.no_grad()
def test_generation_stops_on_eos_and_validates_input():
    source, ids = tiny()
    model = LlamaLogKVForCausalLM(source)
    eos = model(ids, logits_to_keep=1)[0].argmax(-1).item()
    generated, state, reason = model.greedy_generate(ids, eos_token_id=[eos], max_new_tokens=10)
    assert reason == 'eos'
    assert generated.tolist() == [[eos]]
    assert state.offset == ids.shape[1] + 1
    with pytest.raises(ValueError):
        model.greedy_generate(ids.expand(2, -1))
