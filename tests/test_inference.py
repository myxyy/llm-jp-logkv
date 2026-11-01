"""Check that smoke metrics score every next token across chunk boundaries."""
from llm_jp_logkv.settings import load_settings
load_settings()

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_jp_logkv.inference import text_nll
from llm_jp_logkv.llama_model import LlamaLogKVForCausalLM


@pytest.mark.parametrize('mode', ['original', 'dense', 'logkv'])
@torch.no_grad()
def test_chunked_nll_matches_full_sequence(mode):
    torch.manual_seed(22)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8)
    config._attn_implementation = 'eager'
    source = LlamaForCausalLM(config).eval()
    model = source if mode == 'original' else LlamaLogKVForCausalLM(source, mode=mode)
    ids = torch.tensor([[1, 3, 4, 6, 9, 18, 7, 8, 22, 21]])
    logits = source(ids).logits if mode == 'original' else model(ids)[0]
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64),
                                                 ids[:, 1:].reshape(-1), reduction='sum').item()
    result = text_nll(model, ids, mode, chunk_size=3)
    assert result['tokens'] == ids.shape[1] - 1
    assert result['nll_sum'] == pytest.approx(expected, abs=1e-5)
