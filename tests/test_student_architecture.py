"""Student architecture switches, retained histories, and checkpoint compatibility."""
from llm_jp_logkv.settings import load_settings
load_settings()

import json

import pytest
import torch

from llm_jp_logkv import train, training_launcher
from llm_jp_logkv.distillation import DistillationModel, adapter_state, student_from_checkpoint
from llm_jp_logkv.inference import file_hash
from llm_jp_logkv.llama_model import LlamaLogKVForCausalLM
from llm_jp_logkv.logkv.attention import CausalConvBlock
from test_distillation import tiny


def args_for(*flags):
    return train.parse_args(['--run-name', 'architecture-test', '--data', 'unused', *flags])


def test_flags_forwarding_validation_and_legacy_resume():
    defaults = args_for()
    assert not defaults.no_rope and defaults.conv_kernel_size == 0
    legacy = train.model_config(defaults)
    del legacy['no_rope'], legacy['conv_kernel_size'], legacy['attention_training']
    train.check_model_config(legacy, defaults)
    for flags in (['--no-rope'], ['--conv-kernel-size', '4'],
                  ['--attention-training', 'full'],
                  ['--no-rope', '--conv-kernel-size', '4']):
        args = args_for(*flags)
        forwarded = train.parse_args(training_launcher.trainer_arguments(args))
        assert train.model_config(forwarded) == train.model_config(args)
        train.check_model_config(train.model_config(args), args)
        with pytest.raises(ValueError, match='configuration differs'):
            train.check_model_config(legacy, args)
        with pytest.raises(ValueError, match='configuration differs'):
            train.check_model_config(train.model_config(args), defaults)
    with pytest.raises(SystemExit):
        args_for('--conv-kernel-size', '-1')
    with pytest.raises(ValueError, match='nonnegative'):
        LlamaLogKVForCausalLM(tiny(), conv_kernel_size=-1)


def test_no_rope_skips_rotary_in_both_paths_without_changing_teacher(monkeypatch):
    teacher = tiny()
    ids = torch.tensor([[1, 4, 5, 9, 2]])
    expected = teacher(ids).logits.detach()
    model = DistillationModel(teacher, rank=2, no_rope=True)
    attention = model.student.attentions[0]
    hidden = torch.randn(1, 5, 32, dtype=torch.float64)
    q, k, _ = attention._project(hidden, (None, None))
    torch.testing.assert_close(q, attention.q_proj(hidden).view(1, 5, 4, 8).transpose(1, 2))
    torch.testing.assert_close(k, attention.k_proj(hidden).view(1, 5, 2, 8).transpose(1, 2))
    # The teacher's module is still shared, but the disabled student never calls it.
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail('Student invoked RoPE while disabled')
        patch.setattr(teacher.model.rotary_emb, 'forward', forbidden)
        grouped = model.student.lm_head(model.student_hidden(ids))
        streaming, _ = model.student(ids)
        torch.testing.assert_close(grouped, streaming, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(teacher(ids).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize('kernel', [1, 4, 8])
@pytest.mark.parametrize('no_rope', [False, True])
@torch.no_grad()
def test_conv_state_crosses_segments_and_is_immutable(kernel, no_rope):
    model = DistillationModel(tiny(), rank=2, no_rope=no_rope, conv_kernel_size=kernel)
    ids = torch.randint(71, (2, 21))
    full = model.student.lm_head(model.student_hidden(ids))
    chunks = [1, 2, 4, 1, 13]  # Includes splits shorter than the convolution window.
    state, outputs = None, []
    for part in ids.split(chunks, dim=1):
        saved = None if state is None else tuple(t.clone() for t in state.conv_caches)
        logits, new_state = model.student(part, state)
        if saved is not None:
            for a, b in zip(saved, state.conv_caches):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        outputs.append(logits)
        state = new_state
    torch.testing.assert_close(torch.cat(outputs, 1), full, atol=1e-12, rtol=1e-12)
    assert all(t.shape == (2, kernel - 1, 32) for t in state.conv_caches)
    attention_bytes = sum(t.numel()*t.element_size() for s in state.layers
                          for level in s.payload[0] for t in level)
    assert state.tensor_bytes == attention_bytes + 3*2*(kernel-1)*32*8
    other_ids = ids.clone()
    other_ids[:, -1] = (other_ids[:, -1]+1) % 71
    other = model.student.lm_head(model.student_hidden(other_ids))
    torch.testing.assert_close(other[:, :-1], full[:, :-1], atol=0, rtol=0)
    without_conv = LlamaLogKVForCausalLM(tiny())
    with pytest.raises(ValueError, match='convolution configuration'):
        without_conv(ids[:, :1], state)
    with pytest.raises(ValueError, match='convolution configuration'):
        model.student(ids[:, :1], without_conv(ids[:, :1])[1])


def test_conv_matches_unmodified_upstream_forward_step_and_gradients():
    model = DistillationModel(tiny(), rank=2, conv_kernel_size=4)
    adapter = model.student.causal_convs[0]
    reference = CausalConvBlock(32, 4).double()
    reference.load_state_dict(adapter.block.state_dict())
    x = torch.randn(2, 13, 32, dtype=torch.float64, requires_grad=True)
    actual = adapter(x)
    expected = reference(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    a, cache = adapter(x[:, :2], streaming=True)
    b, _ = adapter(x[:, 2:], cache, streaming=True)
    torch.testing.assert_close(torch.cat((a, b), 1), expected, atol=1e-12, rtol=1e-12)
    actual.sum().backward(retain_graph=True)
    expected.sum().backward()
    for p, r in zip(adapter.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad.double(), r.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('legacy,no_rope,kernel,training', [
    (True, False, 0, 'lora'), (False, True, 0, 'lora'),
    (False, False, 4, 'lora'), (False, True, 4, 'lora'),
    (False, False, 0, 'full'), (False, True, 0, 'full'),
    (False, False, 4, 'full'), (False, True, 4, 'full')])
@torch.no_grad()
def test_checkpoint_restores_architecture_and_all_weights(tmp_path, legacy, no_rope, kernel, training):
    args = args_for(*(['--no-rope'] if no_rope else []), '--conv-kernel-size', str(kernel),
                    '--attention-training', training, '--rank', '2', '--alpha', '4')
    config = train.model_config(args)
    if legacy:
        del config['no_rope'], config['conv_kernel_size'], config['attention_training']
    model = DistillationModel(tiny(), rank=2, alpha=4, no_rope=no_rope, conv_kernel_size=kernel,
                              attention_training=training)
    for p in model.student.parameters():
        if p.requires_grad:
            p.add_(torch.randn_like(p)*.01)
    torch.save(adapter_state(model.student), tmp_path/'adapter.pt')
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'complete.json').write_text(json.dumps({'adapter_sha256': file_hash(tmp_path/'adapter.pt')}))
    restored = student_from_checkpoint(tiny(), tmp_path, model_id=config['model_id'],
                                       model_revision=config['model_revision'])
    assert (restored.no_rope, restored.conv_kernel_size) == (no_rope, kernel)
    assert restored.attention_training == training
    ids = torch.tensor([[1, 9, 8, 7, 2]])
    torch.testing.assert_close(restored(ids)[0], model.student(ids)[0], atol=0, rtol=0)
    assert adapter_state(restored).keys() == adapter_state(model.student).keys()
