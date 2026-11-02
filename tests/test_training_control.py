from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from llm_jp_logkv import train


def control_args(root):
    return SimpleNamespace(control_file=root/'control.cmd', run_dir=root,
                           max_hours=8., max_steps=10, lr=1e-4,
                           warmup_steps=50, min_lr_ratio=.1)


@pytest.mark.parametrize('contents', [None, '', '\n', 'pause\n', 'save_and_exit later\n'])
def test_missing_or_unrecognized_command_does_not_stop_or_delete(tmp_path, contents):
    path=tmp_path/'control.cmd'
    if contents is not None:
        path.write_text(contents)
    assert not train.consume_save_and_exit(path)
    if contents is not None:
        assert path.read_text()==contents


def test_save_and_exit_consumed_once(tmp_path):
    path=tmp_path/'control.cmd'
    path.write_text('  save_and_exit\n')
    assert train.consume_save_and_exit(path)
    assert not path.exists()
    assert not train.consume_save_and_exit(path)


@pytest.mark.parametrize('trigger,expected', [
    ('none', None), ('steps', 'max_steps'), ('time', 'deadline'),
    ('STOP', 'stop_request'), ('signal', 'stop_request'),
])
def test_existing_stop_conditions(tmp_path, monkeypatch, trigger, expected):
    monkeypatch.setattr(dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(dist, 'broadcast', lambda tensor, src: None)
    args=control_args(tmp_path)
    if trigger=='STOP':
        (tmp_path/'STOP').touch()
    elapsed,reason,lr=train.training_control(
        args,10 if trigger=='steps' else 2,28800 if trigger=='time' else 120,
        trigger=='signal','cpu')
    assert reason==expected
    assert 0 < lr <= args.lr


def _distributed_control_worker(rank, root):
    root=Path(root)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),
                            rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        args=control_args(root)
        if rank:
            args.control_file=root/'nonleader.cmd'
        # The command wins even when step and time limits are reached. Only
        # rank zero's file and elapsed time may determine the shared decision.
        elapsed,reason,lr=train.training_control(args,10,28800 if rank==0 else 0,False,'cpu')
        assert reason=='save_and_exit'
        assert elapsed==28800
        assert lr>0
    finally:
        dist.destroy_process_group()


def test_save_and_exit_broadcast_to_all_ranks(tmp_path):
    (tmp_path/'control.cmd').write_text('save_and_exit\n')
    (tmp_path/'nonleader.cmd').write_text('save_and_exit\n')
    mp.spawn(_distributed_control_worker,args=(str(tmp_path),),nprocs=2,join=True)
    assert not (tmp_path/'control.cmd').exists()
    assert (tmp_path/'nonleader.cmd').read_text()=='save_and_exit\n'


@pytest.mark.parametrize('custom', [False, True])
def test_control_file_cli_default_and_override(tmp_path, monkeypatch, custom):
    argv=['train','--data',str(tmp_path/'data'),
          '--run-dir',str(train.SETTINGS.paths['EXPERIMENT_DIR']/'control-cli-test')]
    if custom:
        argv+=['--control-file',str(tmp_path/'custom.cmd')]
    monkeypatch.setattr(sys,'argv',argv)
    assert train.parse_args().control_file==(tmp_path/'custom.cmd' if custom else Path('control.cmd'))
