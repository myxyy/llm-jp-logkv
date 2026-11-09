from copy import deepcopy

from llm_jp_logkv.settings import load_settings
load_settings()

import pytest
import torch

from llm_jp_logkv import fsdp_launcher, train_fsdp, training_launcher
from llm_jp_logkv.distillation import DistillationModel, adapter_state
from llm_jp_logkv.remote_fsdp import RemoteStudent
from test_distillation import tiny


@pytest.mark.parametrize('conv', [0, 4])
@pytest.mark.parametrize('no_rope', [False, True])
def test_remote_loss_gradients_and_source_independence(conv, no_rope):
    source = tiny()
    torch.manual_seed(9)
    reference = DistillationModel(source, attention_training='full', conv_kernel_size=conv,
                                  no_rope=no_rope, loss_chunk_tokens=3)
    torch.manual_seed(9)
    remote = RemoteStudent(deepcopy(source), conv_kernel_size=conv, no_rope=no_rope, loss_chunk_tokens=3)
    assert not hasattr(remote, 'teacher')
    assert not any('self_attn' in name for name, _ in remote.named_parameters())
    assert all(hasattr(layer, 'self_attn') for layer in source.model.layers)
    before = {name: p.clone() for name, p in source.named_parameters()}
    ids = torch.tensor([[1, 4, 5, 9, 8, 7, 6, 2], [3, 8, 1, 4, 0, 0, 0, 0]])
    lengths = torch.tensor([8, 4])
    hidden = source.model(ids[:, :-1], use_cache=False).last_hidden_state.detach().requires_grad_()
    expected, stats = reference(ids, lengths)
    actual, actual_stats = remote(ids, lengths, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_stats, stats, rtol=0, atol=0)
    expected.backward()
    actual.backward()
    assert hidden.grad is None
    expected_params = dict(reference.student.named_parameters())
    for name, p in remote.student.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, expected_params[name].grad, rtol=0, atol=0)
        else:
            assert p.grad is None
    assert set(adapter_state(remote.student)) == set(adapter_state(reference.student))
    for name, p in source.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='shape'):
        remote(ids, lengths, hidden[:, :-1])


def test_fsdp_launcher_contract():
    args = fsdp_launcher.parse_args(['--run-name', 'example', '--data', 'unused'])
    assert args.gpus == '1,2,3,4,5' and args.teacher_gpu == '0'
    assert args.attention_training == 'full'
    assert args.loss_chunk_tokens == 64
    assert args.num_epochs == 1 and args.max_hours is None and args.max_checkpoints == 2
    passed = training_launcher.trainer_arguments(args, train_fsdp)
    parsed = train_fsdp.parse_args(passed)
    assert parsed.run_name == args.run_name
    for extra in (['--gpus', '0,1'], ['--teacher-gpu', '1'], ['--gpus', '2,2'],
                  ['--attention-training', 'lora']):
        with pytest.raises(SystemExit):
            fsdp_launcher.parse_args(['--run-name', 'example', '--data', 'unused', *extra])


@pytest.mark.parametrize('count', [1, 4, 5, 7, 11])
def test_validation_padding_never_duplicates_scored_examples(count):
    data = [torch.tensor([i+1, 2, 3]) for i in range(count)]
    observed = []
    forward_counts = []
    scored = 0
    for rank in range(5):
        batches = list(train_fsdp.validation_batches(data, list(range(count)), rank, 5, 2, 8, 0))
        forward_counts.append(len(batches))
        for ids, lengths in batches:
            assert ids.shape == (2, 9)
            observed.extend((ids[lengths > 1, 0]-1).tolist())
            scored += (lengths-1).sum().item()
    assert len(set(forward_counts)) == 1
    assert sorted(observed) == list(range(count))
    assert scored == count*2
