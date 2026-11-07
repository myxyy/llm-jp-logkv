"""Full attention updates must not train or alias the frozen teacher / FFN."""
from llm_jp_logkv.settings import load_settings
load_settings()

import json

import pytest
import torch

from llm_jp_logkv import train, training_launcher
from llm_jp_logkv.distillation import (
    DistillationModel, FullAttentionLinear, adapter_state, load_adapter, student_from_checkpoint,
)
from test_distillation import tiny


def test_full_cli_and_legacy_lora_config():
    base = ['--run-name', 'full-test', '--data', 'unused']
    default = train.parse_args(base)
    assert default.attention_training == 'lora'
    legacy = train.model_config(default)
    del legacy['attention_training']
    train.check_model_config(legacy, default)
    full = training_launcher.parse_args(base + ['--attention-training', 'full'])
    roundtrip = train.parse_args(training_launcher.trainer_arguments(full))
    assert train.model_config(full) == train.model_config(roundtrip)
    for config, args in ((legacy, full), (train.model_config(full), default)):
        with pytest.raises(ValueError, match='configuration differs'):
            train.check_model_config(config, args)
    # LoRA-only settings have no effect on the full-mode model or resume contract.
    other = train.parse_args(base + ['--attention-training', 'full', '--rank', '0', '--alpha', '1'])
    train.check_model_config(train.model_config(full), other)
    with pytest.raises(SystemExit):
        train.parse_args(base + ['--attention-training', 'all'])
    with pytest.raises(ValueError, match='attention_training'):
        DistillationModel(tiny(), attention_training='all')


@pytest.mark.parametrize('bias', [False, True])
def test_full_projection_copies_pretrained_values_without_aliasing(bias):
    base = torch.nn.Linear(8, 4, bias=bias).to(torch.bfloat16).requires_grad_(False)
    projection = FullAttentionLinear(base)
    assert projection.weight.dtype == torch.float32
    assert projection.weight.data_ptr() != base.weight.data_ptr()
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    torch.testing.assert_close(projection(x), base(x), atol=0, rtol=0)
    projection(x).float().square().sum().backward()
    assert projection.weight.grad is not None and base.weight.grad is None
    if bias:
        assert projection.bias.data_ptr() != base.bias.data_ptr()
        assert projection.bias.grad is not None and base.bias.grad is None


@pytest.mark.parametrize('conv_kernel_size', [0, 4])
def test_full_checkpoint_optimizer_resume_and_frozen_modules(tmp_path, monkeypatch, conv_kernel_size):
    teacher = tiny()
    frozen = {name: p.detach().clone() for name, p in teacher.named_parameters()}
    options = dict(attention_training='full', no_rope=True, conv_kernel_size=conv_kernel_size)
    model = DistillationModel(teacher, **options)
    for layer, attention in zip(teacher.model.layers, model.student.attentions):
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            assert getattr(attention, name).weight.data_ptr() != getattr(layer.self_attn, name).weight.data_ptr()
    assert all(not p.requires_grad for p in model.student.layers.parameters())
    assert all(name.startswith(('attentions.', 'causal_convs.')) for name in adapter_state(model.student))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    ids = torch.tensor([[1, 4, 5, 9, 8, 7, 6, 2]])
    lengths = torch.tensor([8])
    def step(network, opt):
        opt.zero_grad(set_to_none=True)
        loss, stats = network(ids, lengths)
        (loss / stats[2]).backward()
        opt.step()
        return loss.detach()
    step(model, optimizer)
    args = train.parse_args(['--data', str(tmp_path/'data'), '--run-name', 'full-test',
                             '--attention-training', 'full', '--no-rope',
                             '--conv-kernel-size', str(conv_kernel_size)])
    args.run_dir = tmp_path/'run'
    args.run_dir.mkdir()
    args.data.mkdir()
    (args.data/'manifest.json').write_text('{}')
    args.run_start_unix = 1234.
    monkeypatch.setitem(train.SETTINGS.paths, 'CHECKPOINT_DIR', tmp_path/'checkpoints')
    monkeypatch.setattr(torch.cuda, 'get_rng_state', torch.get_rng_state)
    monkeypatch.setattr(train.dist, 'gather_object', lambda local, gathered, dst: gathered.__setitem__(0, local))
    monkeypatch.setattr(train.dist, 'barrier', lambda: None)
    path = train.save_checkpoint(model, optimizer, args, 1, 0, 1, 2., 1, 0)
    saved_config = json.loads((path/'config.json').read_text())
    train.check_model_config(saved_config, args)
    saved = torch.load(path/'training_state.pt', weights_only=True)
    restored = DistillationModel(tiny(), **options)
    load_adapter(restored.student, torch.load(path/'adapter.pt', weights_only=True))
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.001)
    restored_optimizer.load_state_dict(saved['optimizer'])
    inferred = student_from_checkpoint(tiny(), path, model_id=train.SETTINGS.model_id,
                                       model_revision=train.SETTINGS.model_revision)
    torch.testing.assert_close(inferred(ids)[0], model.student(ids)[0], atol=0, rtol=0)
    torch.testing.assert_close(step(model, optimizer), step(restored, restored_optimizer), atol=0, rtol=0)
    for name, value in adapter_state(model.student).items():
        torch.testing.assert_close(value, adapter_state(restored.student)[name], atol=0, rtol=0)
    for name, p in teacher.named_parameters():
        assert not p.requires_grad and p.grad is None
        torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
