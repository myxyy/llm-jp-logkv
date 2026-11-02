from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from llm_jp_logkv import train


def control_args(root):
    return SimpleNamespace(control_file=root/'control.cmd', run_dir=root,
                           max_hours=8., num_epochs=0, max_steps=10, lr=1e-4,
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
        trigger=='signal','cpu',epoch=0,batch_in_epoch=0,batches_per_epoch=1)
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
        elapsed,reason,lr=train.training_control(args,10,28800 if rank==0 else 0,False,'cpu',
                                               epoch=0,batch_in_epoch=0,batches_per_epoch=1)
        assert reason=='save_and_exit'
        assert elapsed==28800
        assert lr>0
        args.max_hours=None; args.num_epochs=1; args.max_steps=0
        _,reason,_=train.training_control(args,2,43200,False,'cpu',
                                          epoch=0,batch_in_epoch=4,batches_per_epoch=5)
        assert reason is None  # Twelve hours does not stop the default mode.
        _,reason,_=train.training_control(args,3,43200,False,'cpu',
                                          epoch=1 if rank==0 else 0,batch_in_epoch=0,batches_per_epoch=5)
        assert reason=='num_epochs'
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
          '--run-name','control-cli-test']
    if custom:
        argv+=['--control-file',str(tmp_path/'custom.cmd')]
    monkeypatch.setattr(sys,'argv',argv)
    assert train.parse_args().control_file==(tmp_path/'custom.cmd' if custom else Path('control.cmd'))


@pytest.mark.parametrize('flags,expected', [
    ([], (None, 1, 0)),
    (['--num-epochs','3'], (None, 3, 0)),
    (['--max-hours','8'], (8., 1, 0)),
    (['--max-hours','8','--num-epochs','0'], (8., 0, 0)),
    (['--max-steps','20','--num-epochs','0'], (None, 0, 20)),
])
def test_explicit_limits_and_unlimited_time_default(monkeypatch, flags, expected):
    monkeypatch.setattr(sys,'argv',['train','--data','unused',
        '--run-name','epoch-cli-test',*flags])
    args=train.parse_args()
    assert (args.max_hours,args.num_epochs,args.max_steps)==expected


@pytest.mark.parametrize('flags', [
    ['--max-hours','0'], ['--max-hours','-1'], ['--max-hours','nan'], ['--max-hours','inf'],
    ['--num-epochs','-1'], ['--num-epochs','0'], ['--max-steps','-1'],
])
def test_invalid_limits_rejected(monkeypatch, flags):
    monkeypatch.setattr(sys,'argv',['train','--data','unused',
        '--run-name','epoch-cli-test',*flags])
    with pytest.raises(SystemExit):
        train.parse_args()


def test_unlimited_time_schedule_uses_epoch_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(dist,'get_rank',lambda:0)
    monkeypatch.setattr(dist,'broadcast',lambda tensor,src:None)
    args=control_args(tmp_path)
    args.max_hours=None; args.num_epochs=1; args.max_steps=0; args.warmup_steps=0
    def control(elapsed,batch,epoch=0):
        return train.training_control(args,10,elapsed,False,'cpu',
                                      epoch=epoch,batch_in_epoch=batch,batches_per_epoch=4)
    assert control(1,0)[1] is None
    assert control(43200,0)[1] is None
    assert control(1,0)[2]==control(43200,0)[2]
    assert control(43200,2)[2]==pytest.approx(args.lr*.55)
    assert control(43200,0,epoch=1)[1]=='num_epochs'


def make_cursor(epoch=0, batch=0):
    dataset=torch.arange(5)
    sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=False)
    loader=DataLoader(dataset,batch_size=1,sampler=sampler)
    return train.EpochBatchIterator(loader,sampler,epoch,batch)


@pytest.mark.parametrize('grad_accum,expected_sizes', [(1,[1]*5),(4,[4,1]),(8,[5])])
def test_one_epoch_consumes_remainder_without_starting_second_epoch(grad_accum, expected_sizes):
    cursor=make_cursor()
    updates=[]
    while cursor.epoch<1:
        updates.append(cursor.next_update(grad_accum))
    assert [len(x) for x in updates]==expected_sizes
    assert [b.item() for update in updates for b in update]==list(range(5))
    assert cursor.epoch==1 and cursor.batch_in_epoch==0


def test_resume_finishes_only_the_remaining_batches_and_can_extend_epochs():
    cursor=make_cursor()
    assert [x.item() for x in cursor.next_update(4)]==[0,1,2,3]
    resumed=make_cursor(cursor.epoch,cursor.batch_in_epoch)
    assert [x.item() for x in resumed.next_update(4)]==[4]
    assert (resumed.epoch,resumed.batch_in_epoch)==(1,0)
    # Only a subsequent update enters epoch two; no first-epoch update crosses.
    assert [x.item() for x in resumed.next_update(4)]==[0,1,2,3]
    assert [x.item() for x in resumed.next_update(4)]==[4]
    assert (resumed.epoch,resumed.batch_in_epoch)==(2,0)


def test_legacy_cursor_at_epoch_end_is_normalized():
    cursor=make_cursor(epoch=0,batch=5)
    assert (cursor.epoch,cursor.batch_in_epoch)==(1,0)


def test_legacy_resume_can_remove_time_limit_but_not_change_data_order(monkeypatch):
    monkeypatch.setattr(sys,'argv',['train','--data','unused',
        '--run-name','epoch-cli-test'])
    args=train.parse_args()
    old_signature={**train.training_signature(args),'max_hours':8.}
    train.check_resume_signature(old_signature,args)
    args.num_epochs=3; args.max_hours=12
    train.check_resume_signature(old_signature,args)
    args.grad_accum+=1
    with pytest.raises(ValueError,match='configuration differs'):
        train.check_resume_signature(old_signature,args)
