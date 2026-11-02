import json
from types import SimpleNamespace
import sys

import pytest
import torch

from llm_jp_logkv import train


@pytest.mark.parametrize('flags,expected', [([],2), (['--max-checkpoints','1'],1), (['--max-checkpoints','5'],5)])
def test_checkpoint_retention_cli(monkeypatch, flags, expected):
    monkeypatch.setattr(sys,'argv',['train','--data','unused',
        '--run-dir',str(train.SETTINGS.paths['EXPERIMENT_DIR']/'retention-cli-test'),*flags])
    assert train.parse_args().max_checkpoints==expected


@pytest.mark.parametrize('value', ['0','-1','1.5'])
def test_invalid_checkpoint_retention_cli(monkeypatch, value):
    monkeypatch.setattr(sys,'argv',['train','--data','unused',
        '--run-dir',str(train.SETTINGS.paths['EXPERIMENT_DIR']/'retention-cli-test'),
        '--max-checkpoints',value])
    with pytest.raises(SystemExit):
        train.parse_args()


@pytest.fixture
def saver(tmp_path, monkeypatch):
    monkeypatch.setitem(train.SETTINGS.paths,'EXPERIMENT_DIR',tmp_path/'experiments')
    monkeypatch.setitem(train.SETTINGS.paths,'CHECKPOINT_DIR',tmp_path/'checkpoints')
    monkeypatch.setattr(sys,'argv',['train','--data',str(tmp_path/'data'),
        '--run-dir',str(tmp_path/'experiments'/'run')])
    args=train.parse_args()
    args.run_start_unix=1234.
    args.run_dir.mkdir(parents=True)
    args.data.mkdir()
    (args.data/'manifest.json').write_text('{}')
    model=SimpleNamespace(student=torch.nn.Linear(2,1))
    optimizer=torch.optim.AdamW(model.student.parameters())
    model.student(torch.ones(1,2)).sum().backward()
    optimizer.step()
    monkeypatch.setattr(train,'adapter_state',lambda student:student.state_dict())
    monkeypatch.setattr(torch.cuda,'get_rng_state',torch.get_rng_state)
    def gather(local, gathered, dst):
        if gathered is not None:
            gathered[0]=local
    monkeypatch.setattr(train.dist,'gather_object',gather)
    monkeypatch.setattr(train.dist,'barrier',lambda:None)
    def save(step, rank=0):
        return train.save_checkpoint(model,optimizer,args,step,1,3,20.,1,rank)
    return SimpleNamespace(args=args,model=model,optimizer=optimizer,save=save,
                           root=train.SETTINGS.paths['CHECKPOINT_DIR']/args.run_dir.name)


@pytest.mark.parametrize('keep', [1,2,5])
def test_saves_keep_latest_steps_and_preserve_resumable_state(saver, keep):
    saver.args.max_checkpoints=keep
    # Cross the seven-digit boundary to catch lexical rather than numeric sorting.
    steps=[2,10,9999999,10000000]
    paths=[saver.save(step) for step in steps]
    assert [p for p in paths if p.exists()]==paths[-keep:]
    latest=paths[-1]
    complete=json.loads((latest/'complete.json').read_text())
    assert complete['adapter_sha256']==train.file_hash(latest/'adapter.pt')
    assert complete['training_state_sha256']==train.file_hash(latest/'training_state.pt')
    state=torch.load(latest/'training_state.pt',weights_only=True)
    assert (state['step'],state['epoch'],state['batch_in_epoch'])==(steps[-1],1,3)
    assert state['run_start_unix']==saver.args.run_start_unix
    assert state['rngs'][0]['cpu'].equal(torch.get_rng_state())
    restored_optimizer=torch.optim.AdamW(saver.model.student.parameters())
    restored_optimizer.load_state_dict(state['optimizer'])
    for parameter,expected in saver.optimizer.state.items():
        for key,value in expected.items():
            torch.testing.assert_close(restored_optimizer.state[parameter][key],value)
    train.check_resume_signature(state['training_signature'],saver.args)
    prune_log=saver.args.run_dir/'checkpoint-pruning.jsonl'
    pruned=[json.loads(line)['step'] for line in prune_log.read_text().splitlines()] if prune_log.exists() else []
    assert pruned==steps[:-keep]


def test_repeated_final_save_does_not_rewrite_checkpoint_and_can_lower_retention(saver):
    old=saver.save(1)
    final=saver.save(2)
    original={p.name:p.read_bytes() for p in final.iterdir()}
    signature=train.training_signature(saver.args)
    saver.args.max_checkpoints=1
    train.check_resume_signature(signature,saver.args)
    assert saver.save(2)==final
    assert not old.exists()
    assert {p.name:p.read_bytes() for p in final.iterdir()}==original
    saver.save(2)  # The usual final save at an already saved periodic boundary.
    assert len((saver.args.run_dir/'checkpoint-pruning.jsonl').read_text().splitlines())==1


@pytest.mark.parametrize('failed_file', ['adapter.pt','training_state.pt','config.json','complete.json'])
def test_failed_save_never_prunes_previous_checkpoints(saver, monkeypatch, failed_file):
    old=saver.save(1)
    saver.args.max_checkpoints=1
    original_save=train.torch.save
    original_json=train.write_json
    def save(value,path):
        if path.name==failed_file:
            raise OSError('simulated failed write')
        original_save(value,path)
    def write_json(path,value):
        if path.name==failed_file:
            raise OSError('simulated failed write')
        original_json(path,value)
    monkeypatch.setattr(train.torch,'save',save)
    monkeypatch.setattr(train,'write_json',write_json)
    with pytest.raises(OSError,match='simulated failed write'):
        saver.save(2)
    assert (old/'complete.json').is_file()
    assert not (saver.args.run_dir/'checkpoint-pruning.jsonl').exists()


def test_pruning_leaves_incomplete_unrecognized_foreign_and_symlink_paths(saver):
    old=saver.save(10)
    ignored=[]
    for name,metadata in [
        ('step-0000001',None),
        ('step-0000002','{'),
        ('step-0000003','[]'),
        ('step-0000004',json.dumps({'step':4,'source_run':'/unrelated/run'})),
        ('step-0000005',json.dumps({'step':999,'source_run':str(saver.args.run_dir)})),
        ('step-6.tmp',None),
        ('notes',None),
    ]:
        path=saver.root/name
        path.mkdir()
        for filename in ('adapter.pt','training_state.pt','config.json'):
            (path/filename).write_text('untouched')
        if metadata is not None:
            (path/'complete.json').write_text(metadata)
        ignored.append(path)
    symlink=saver.root/'step-0000000'
    symlink.symlink_to(old,target_is_directory=True)
    regular_file=saver.root/'step-0000007'
    regular_file.write_text('untouched')
    saver.args.max_checkpoints=1
    latest=saver.save(20)
    assert latest.is_dir() and not old.exists()
    assert all((path/'adapter.pt').read_text()=='untouched' for path in ignored)
    assert symlink.is_symlink()
    assert regular_file.read_text()=='untouched'


def test_resumed_run_keeps_source_checkpoints(saver):
    sources=[saver.save(1),saver.save(2)]
    saver.args.resume=sources[-1]
    saver.args.run_dir=saver.args.run_dir.with_name('resumed')
    saver.args.run_dir.mkdir()
    saver.args.max_checkpoints=1
    first=saver.save(3)
    latest=saver.save(4)
    assert not first.exists() and latest.exists()
    assert all((path/'complete.json').is_file() for path in sources)


def test_existing_checkpoint_from_different_run_is_not_reused_or_pruned(saver):
    existing=saver.save(1)
    # The current layout uses run_dir.name, so nested runs could share a basename.
    saver.args.run_dir=saver.args.run_dir.parent/'nested'/saver.args.run_dir.name
    saver.args.run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='another run'):
        saver.save(1)
    assert (existing/'complete.json').is_file()


def test_nonleader_only_participates_in_collectives(saver, monkeypatch):
    events=[]
    monkeypatch.setattr(train.dist,'gather_object',lambda *args,**kwargs:events.append('gather'))
    monkeypatch.setattr(train.dist,'barrier',lambda:events.append('barrier'))
    def unexpected(*args):
        pytest.fail('Nonleader must not prune checkpoints')
    monkeypatch.setattr(train,'prune_checkpoints',unexpected)
    saver.save(1,rank=1)
    assert events==['gather','barrier']
    assert not saver.root.exists()
