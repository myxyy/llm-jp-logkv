from llm_jp_logkv.settings import load_settings
load_settings()

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_jp_logkv.distillation import DistillationModel, adapter_state, load_adapter
from llm_jp_logkv.llama_model import LlamaLogKVForCausalLM


def tiny():
    torch.manual_seed(19)
    config = LlamaConfig(vocab_size=71, hidden_size=32, intermediate_size=56,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8)
    config._attn_implementation = 'eager'
    return LlamaForCausalLM(config).double().eval()


def test_grouped_adapter_output_and_gradients_match_streaming_path():
    source = tiny()
    student = LlamaLogKVForCausalLM(source)
    x = torch.randn(2, 19, 32, dtype=torch.float64, requires_grad=True)
    rope = source.model.rotary_emb(x, torch.arange(19)[None])
    attention = student.attentions[0]
    grouped = attention.forward_full(x, rope)
    stream, _ = attention(x, rope)
    torch.testing.assert_close(grouped, stream, atol=1e-12, rtol=1e-12)
    params = [x, *attention.parameters()]
    grads1 = torch.autograd.grad(grouped.square().sum(), params, retain_graph=True)
    grads2 = torch.autograd.grad(stream.square().sum(), params)
    for a,b in zip(grads1, grads2):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('no_rope,conv_kernel_size', [(False, 0), (True, 0), (False, 4), (True, 4)])
@pytest.mark.parametrize('attention_training', ['lora', 'full'])
def test_training_changes_only_student_and_roundtrips_adapter(no_rope, conv_kernel_size, attention_training):
    source = tiny()
    ids = torch.tensor([[1,4,5,9,10,8,7,6,2]])
    expected_teacher = source(ids).logits.detach().clone()
    options = dict(no_rope=no_rope, conv_kernel_size=conv_kernel_size)
    initial_student = LlamaLogKVForCausalLM(source, **options)(ids)[0].detach()
    options['attention_training'] = attention_training
    train = DistillationModel(source, rank=2, alpha=4, loss_chunk_tokens=3, **options)
    if not conv_kernel_size:  # Added convolutions have upstream random initialization.
        torch.testing.assert_close(train.student(ids)[0], initial_student, atol=0,rtol=0)
    optimizer = torch.optim.AdamW([p for p in train.parameters() if p.requires_grad], lr=.01)
    before = adapter_state(train.student)
    for _ in range(2):
        loss,stats = train(ids, torch.tensor([9]))
        (loss/stats[2]).backward()
        grads = [p.grad for p in train.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    after = adapter_state(train.student)
    assert any(not torch.equal(before[k],after[k]) for k in before)
    if attention_training == 'full':
        assert not any('lora_' in name for name in before)
        attention_names = [k for k in before if k.startswith('attentions.')]
        assert len(attention_names) == 12  # All four matrices in every layer.
        assert all(not torch.equal(before[k], after[k]) for k in attention_names)
    if conv_kernel_size:
        conv_names = [k for k in before if k.startswith('causal_convs.')]
        assert len(conv_names) == 9  # RMSNorm weight + convolution weight/bias, per layer.
        assert all(not torch.equal(before[k], after[k]) for k in conv_names)
        assert all(after[k].dtype == torch.float32 for k in conv_names)
    assert all(p.grad is None and not p.requires_grad for p in source.parameters())
    torch.testing.assert_close(source(ids).logits, expected_teacher, atol=0,rtol=0)
    restored = DistillationModel(tiny(),rank=2,alpha=4,**options)
    load_adapter(restored.student, after)
    torch.testing.assert_close(restored.student(ids)[0], train.student(ids)[0],atol=0,rtol=0)


@pytest.mark.parametrize('dtype', [torch.float64, torch.bfloat16])
@pytest.mark.parametrize('no_rope,conv_kernel_size', [(False, 0), (True, 0), (False, 4), (True, 4)])
@pytest.mark.parametrize('attention_training', ['lora', 'full'])
@torch.no_grad()
def test_grouped_full_model_matches_streaming_inference(dtype, no_rope, conv_kernel_size, attention_training):
    model = DistillationModel(tiny().to(dtype), rank=2, alpha=4,
                              no_rope=no_rope, conv_kernel_size=conv_kernel_size,
                              attention_training=attention_training)
    for name, p in model.student.named_parameters():
        if 'lora_B' in name:
            p.normal_(std=.02)
    ids = torch.randint(71, (1, 19))
    grouped = model.student.lm_head(model.student_hidden(ids))
    prefix, state = model.student(ids[:, :7])
    suffix, _ = model.student(ids[:, 7:], state)
    streaming = torch.cat((prefix, suffix), dim=1)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-12
    torch.testing.assert_close(grouped, streaming, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('conv_kernel_size', [0, 4])
@pytest.mark.parametrize('attention_training', ['lora', 'full'])
def test_chunked_full_vocab_loss_checkpointing_and_padding_gradients(conv_kernel_size, attention_training):
    ids = torch.tensor([[1,4,5,9,10,8,7,6,2],[1,9,7,2,0,0,0,0,0]])
    lengths = torch.tensor([9,4])
    torch.manual_seed(33)
    a = DistillationModel(tiny(),rank=2,alpha=4,loss_chunk_tokens=3,checkpoint_layers=True,
                         attention_training=attention_training,
                         conv_kernel_size=conv_kernel_size, no_rope=bool(conv_kernel_size))
    b = DistillationModel(tiny(),rank=2,alpha=4,loss_chunk_tokens=100,checkpoint_layers=False,
                         attention_training=attention_training,
                         conv_kernel_size=conv_kernel_size, no_rope=bool(conv_kernel_size))
    load_adapter(b.student,adapter_state(a.student))
    la,sa=a(ids,lengths); lb,sb=b(ids,lengths)
    torch.testing.assert_close(sa,sb,atol=1e-5,rtol=1e-6)
    assert sa[2].item()==11
    la.backward(); lb.backward()
    for (_,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
        if pa.requires_grad:
            torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=1e-5)
    with torch.no_grad():
        single=sum(a(ids[i:i+1,:lengths[i]],lengths[i:i+1])[0] for i in range(2))
    torch.testing.assert_close(single,la,atol=1e-5,rtol=1e-6)


def test_public_sft_conversion_keeps_analysis_and_final_distinct():
    from llm_jp_logkv.training_data import first_exchange
    raw = [
        {'role':'system','content':[{'type':'system_content','model_identity':'LLM-jp',
                                    'reasoning_effort':'Low','conversation_start_date':'2026-11-02'}]},
        {'role':'user','content':[{'type':'text','text':'17-8+5?'}]},
        {'role':'assistant','channel':'analysis','content':[{'type':'text','text':'17-8=9; 9+5=14'}]},
        {'role':'assistant','channel':'final','content':[{'type':'text','text':'14'}]},
        {'role':'user','content':[{'type':'text','text':'This later turn is intentionally excluded.'}]}]
    messages,kwargs=first_exchange(raw)
    assert messages==[{'role':'user','content':'17-8+5?'},
                      {'role':'assistant','content':'14','thinking':'17-8=9; 9+5=14'}]
    assert kwargs['model_identity']=='LLM-jp' and kwargs['reasoning_effort']=='Low'
    with pytest.raises(ValueError):
        first_exchange([{'role':'tool','content':[{'type':'text','text':'output'}]}])


@pytest.mark.parametrize('temperature', [1., 2.])
def test_distillation_loss_and_gradient_against_direct_full_vocab_reference(temperature):
    import torch.nn.functional as F
    model = DistillationModel(tiny(), rank=2, temperature=temperature)
    student = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student)
    targets = torch.randint(71, (2, 5))
    mask = torch.tensor([[True]*5, [True, True, False, False, False]])
    actual = model._loss_chunk(student, teacher, targets, mask)
    teacher_logits = model.student.lm_head(teacher).float()
    student_logits = model.student.lm_head(student).float()
    kl = F.kl_div(F.log_softmax(student_logits/temperature, -1),
                  F.softmax(teacher_logits/temperature, -1), reduction='none').sum(-1)
    ce = F.cross_entropy(student_logits.flatten(0, 1), targets.flatten(),
                         reduction='none').reshape_as(targets)
    expected = torch.stack(((kl*mask).sum()*temperature**2, (ce*mask).sum()))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual.sum(), student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), student)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-5)


def test_uint32_cache_preserves_large_token_ids_and_complete_conversations(tmp_path):
    import numpy as np
    from llm_jp_logkv.training_data import TokenCache, collate_tokens
    np.array([1, 196607, 2, 1, 70000, 8, 9, 2], dtype=np.uint32).tofile(tmp_path/'tokens.bin')
    np.save(tmp_path/'offsets.npy', np.array([0, 3, 8], dtype=np.uint64))
    np.save(tmp_path/'train.npy', np.array([0, 1], dtype=np.int64))
    cache = TokenCache(tmp_path, 'train', sequence_length=3)
    assert len(cache) == 1  # The longer exchange is excluded, never truncated.
    ids, lengths = collate_tokens([cache[0]], sequence_length=3, pad_token_id=0)
    assert ids.tolist() == [[1, 196607, 2, 0]]
    assert lengths.tolist() == [3]
