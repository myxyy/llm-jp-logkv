"""Offline CPU smoke for the pinned Llama/GQA/cache/LoRA dependency stack."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


def test_tiny_llama_gqa_cache_and_lora():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    ids = torch.tensor([[1, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        baseline = model(ids, use_cache=False).logits
        prefix = model(ids[:, :-1], use_cache=True)
        last = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits
    torch.testing.assert_close(last, baseline[:, -1:], atol=1e-6, rtol=1e-5)
    student = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                            task_type="CAUSAL_LM"))
    student.eval()
    output = student(ids, labels=ids, use_cache=False)
    torch.testing.assert_close(output.logits, baseline, atol=1e-6, rtol=1e-5)
    output.loss.backward()
    grads = [p.grad for name, p in student.named_parameters() if "lora_B" in name]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max() > 0 for g in grads)
