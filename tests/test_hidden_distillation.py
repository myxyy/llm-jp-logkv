"""Independent streaming reference for block-output supervision and gradients."""
from copy import deepcopy

from llm_jp_logkv.settings import load_settings
load_settings()

import pytest
import torch

from llm_jp_logkv import train, train_fsdp, training_launcher
from llm_jp_logkv.distillation import DistillationModel, teacher_hidden_states, loss_metrics
from llm_jp_logkv.remote_fsdp import RemoteStudent
from test_distillation import tiny


def capture_residual_outputs(model, forward):
    # Observe the residual input to post-attention norm and add the MLP output.
    # This independently captures block outputs in teacher AND streaming student.
    values, residuals, handles = [], {}, []
    for i, layer in enumerate(model.layers):
        def before(module, args, i=i):
            residuals[i] = args[0]
        def after(module, args, output, i=i):
            values.append(residuals[i] + output)
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(before))
        handles.append(layer.mlp.register_forward_hook(after))
    try:
        result = forward()
    finally:
        for handle in handles:
            handle.remove()
    return result, torch.stack(values)


@pytest.mark.parametrize('mode', ['lora', 'full'])
@pytest.mark.parametrize('conv', [0, 4])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_mse_matches_streaming_reference_gradients_and_padding(mode, conv, checkpointed):
    model = DistillationModel(tiny(), rank=2, alpha=4, attention_training=mode,
        no_rope=bool(conv), conv_kernel_size=conv, hidden_mse_weight=.3,
        loss_chunk_tokens=3, checkpoint_layers=checkpointed)
    reference = deepcopy(model)
    ids = torch.tensor([[1, 3, 4, 6, 9, 2], [1, 8, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    lengths = torch.tensor([6, 3, 1])  # Includes an entirely unscored FSDP dummy row.
    inputs, targets = ids[:, :-1], ids[:, 1:]
    mask = torch.arange(5)[None] < lengths[:, None]-1
    with torch.no_grad():
        teacher_result, teacher_blocks = capture_residual_outputs(reference.teacher.model,
            lambda: reference.teacher.model(inputs, use_cache=False))
    (_, _), student_blocks = capture_residual_outputs(reference.student,
        lambda: reference.student(inputs))
    expected_mse = ((student_blocks.float()-teacher_blocks.float()).square().mean(-1)*mask).sum((1, 2))
    expected_energy = (teacher_blocks.float().square().mean(-1)*mask).sum((1, 2))
    expected_vocab = reference._loss_chunk(reference.student.norm(student_blocks[-1]),
        teacher_result.last_hidden_state, targets, mask)
    expected = expected_vocab[0] + reference.ce_weight*expected_vocab[1] + .3*expected_mse.mean()
    actual, stats = model(ids, lengths)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(stats[3:6], expected_mse.detach(), atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(stats[6:], expected_energy, atol=1e-7, rtol=2e-6)
    assert stats[2] == 7
    actual.backward(); expected.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
        else:
            assert p.grad is None
    assert all(p.grad is None for p in model.teacher.parameters())
    with torch.no_grad():
        unpadded = sum(model(ids[i:i+1, :lengths[i]], lengths[i:i+1])[0] for i in range(2))
    torch.testing.assert_close(actual, unpadded, atol=2e-6, rtol=2e-6)
    metrics = loss_metrics(stats.double(), model.ce_weight, model.hidden_mse_weight)
    assert metrics['loss'] == pytest.approx(actual.item()/7, rel=2e-6)
    assert metrics['hidden_mse'] == pytest.approx(expected_mse.mean().item()/7, rel=2e-6)
    assert len(metrics['teacher_rms_by_layer']) == 3


def test_teacher_capture_is_pre_final_norm_and_hooks_removed_on_error(monkeypatch):
    source = tiny()
    source.model.norm.weight.data.fill_(2.)
    ids = torch.tensor([[1, 2, 3]])
    expected, blocks = capture_residual_outputs(source.model, lambda: source.model(ids))
    actual = teacher_hidden_states(source, ids, include_blocks=True)
    torch.testing.assert_close(actual['blocks'], blocks, atol=0, rtol=0)
    torch.testing.assert_close(actual['final'], expected.last_hidden_state, atol=0, rtol=0)
    assert not torch.allclose(actual['blocks'][-1], actual['final'])
    assert not actual['blocks'].requires_grad
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(source.model.layers[1], 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        teacher_hidden_states(source, ids, include_blocks=True)
    assert all(not layer._forward_hooks for layer in source.model.layers)


def test_zero_weight_preserves_old_path_and_stats(monkeypatch):
    default = DistillationModel(tiny(), rank=2)
    explicit = DistillationModel(tiny(), rank=2, hidden_mse_weight=0.)
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled loss must not collect block states')
    for layer in explicit.teacher.model.layers:
        monkeypatch.setattr(layer, 'register_forward_hook', forbidden)
    ids, lengths = torch.tensor([[1, 4, 7, 2]]), torch.tensor([4])
    a, sa = default(ids, lengths); b, sb = explicit(ids, lengths)
    assert sa.numel() == sb.numel() == explicit.stats_size == 3
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    a.backward(); b.backward()
    for p, q in zip(default.parameters(), explicit.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_remote_hidden_mse_matches_local_and_detaches_teacher():
    source = tiny()
    local = DistillationModel(source, attention_training='full', hidden_mse_weight=.2)
    remote = RemoteStudent(deepcopy(source), hidden_mse_weight=.2)
    ids, lengths = torch.tensor([[1, 8, 5, 2]]), torch.tensor([4])
    teacher = teacher_hidden_states(source, ids[:, :-1], include_blocks=True)
    for value in teacher.values():
        value.requires_grad_()
    a, sa = local(ids, lengths); b, sb = remote(ids, lengths, teacher)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)
    a.backward(); b.backward()
    local_params = dict(local.student.named_parameters())
    for name, p in remote.student.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, local_params[name].grad, atol=0, rtol=0)
    assert all(value.grad is None for value in teacher.values())
    with pytest.raises(ValueError, match='all block'):
        remote(ids, lengths, teacher['final'])
    with pytest.raises(ValueError, match='shape'):
        remote(ids, lengths, {**teacher, 'blocks': teacher['blocks'][:-1]})


@pytest.mark.parametrize('trainer', [train, train_fsdp])
def test_cli_roundtrip_resume_and_legacy_defaults(trainer):
    base = ['--run-name', 'hidden-mse-test', '--data', 'unused']
    args = trainer.parse_args(base)
    old = train.training_signature(args)
    old.pop('hidden_mse_weight')
    train.check_resume_signature(old, args)
    enabled = trainer.parse_args([*base, '--hidden-mse-weight', '.2'])
    roundtrip = trainer.parse_args(training_launcher.trainer_arguments(enabled, trainer))
    assert roundtrip.hidden_mse_weight == .2
    train.check_resume_signature(train.training_signature(enabled), roundtrip)
    with pytest.raises(ValueError, match='configuration differs'):
        train.check_resume_signature(old, enabled)
    for bad in ['-1', 'nan', 'inf']:
        with pytest.raises(SystemExit):
            trainer.parse_args([*base, '--hidden-mse-weight', bad])
        with pytest.raises(ValueError, match='finite and nonnegative'):
            DistillationModel(tiny(), hidden_mse_weight=float(bad))
