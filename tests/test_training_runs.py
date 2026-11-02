import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llm_jp_logkv import train, training_launcher as launcher
from llm_jp_logkv.training_runs import (completed_checkpoints, create_session, exclusive_run,
                                      named_directory, resolve_resume)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for key, suffix in [('EXPERIMENT_DIR','experiments'), ('CHECKPOINT_DIR','checkpoints'), ('LOG_DIR','logs')]:
        monkeypatch.setitem(train.SETTINGS.paths,key,tmp_path/suffix)
    return train.SETTINGS


def arguments(name='test', resume=None):
    return train.parse_args(['--run-name',name,'--data','unused']+(['--resume',str(resume)] if resume else []))


def checkpoint(settings, args, step):
    root=settings.paths['CHECKPOINT_DIR']/args.run_name
    path=root/f'step-{step:07d}'
    path.mkdir(parents=True)
    for name in ('adapter.pt','training_state.pt','config.json'):
        (path/name).write_text('{}')
    (path/'complete.json').write_text(json.dumps({'step':step,'source_run':str(args.run_dir)}))
    return path


@pytest.mark.parametrize('name', ['', '.', '..', '../escape', '/absolute', 'nested/run', 'nested\\run'])
def test_run_name_rejects_paths(settings, name):
    with pytest.raises(SystemExit):
        arguments(name)


def test_paths_derived_from_name_and_old_options_removed(settings):
    args=arguments('named')
    assert args.run_dir==settings.paths['EXPERIMENT_DIR']/'named'
    for option in ('--run-dir','--log-dir'):
        with pytest.raises(SystemExit):
            launcher.parse_args(['--run-name','named','--data','unused',option,'/unused'])
    root=settings.paths['LOG_DIR']
    root.mkdir()
    (root/'linked').symlink_to(root.parent/'elsewhere')
    with pytest.raises(SystemExit):
        arguments('linked')


def test_launcher_accepts_flat_options_and_old_separator(settings):
    for separator in ([], ['--']):
        args=launcher.parse_args(['--gpus','0,1','--evaluate-final',*separator,
            '--run-name','named','--data','unused','--resume','latest','--max-checkpoints','3'])
        forwarded=train.parse_args(launcher.trainer_arguments(args))
        assert forwarded.run_name=='named' and forwarded.resume=='latest'
        assert forwarded.max_checkpoints==3 and forwarded.max_hours is None
        assert forwarded.num_epochs==1


def test_latest_ignores_incomplete_and_uses_numeric_steps(settings):
    args=arguments(resume='latest')
    older=checkpoint(settings,args,9999999)
    latest=checkpoint(settings,args,10000000)
    incomplete=checkpoint(settings,args,10000001)
    (incomplete/'complete.json').write_text('{')
    missing=checkpoint(settings,args,10000002)
    (missing/'training_state.pt').unlink()
    (latest.parent/'step-10000003').symlink_to(latest,target_is_directory=True)
    assert resolve_resume(args,settings)==latest
    assert completed_checkpoints(latest.parent)==[older,latest]
    args.resume=latest.name
    assert resolve_resume(args,settings)==latest


def test_resume_missing_fresh_restart_and_rewind_are_errors(settings):
    args=arguments(resume='latest')
    with pytest.raises(FileNotFoundError,match='No completed'):
        resolve_resume(args,settings)
    older=checkpoint(settings,args,1)
    latest=checkpoint(settings,args,2)
    args.resume=None
    with pytest.raises(FileExistsError,match='--resume latest'):
        resolve_resume(args,settings)
    args.resume=older
    with pytest.raises(ValueError,match='other/later'):
        resolve_resume(args,settings)
    fork=arguments('fork',older)
    assert resolve_resume(fork,settings)==older
    assert latest.exists()


def test_legacy_run_and_session_files_are_preserved(settings):
    args=arguments(resume='latest')
    args.run_dir.mkdir(parents=True)
    (args.run_dir/'manifest.json').write_text('legacy manifest')
    (args.run_dir/'completed.json').write_text('legacy completion')
    saved=checkpoint(settings,args,1)
    assert resolve_resume(args,settings)==saved
    first=create_session(args.run_dir,'first')
    (first/'completed.json').write_text('first completion')
    second=create_session(args.run_dir,'second')
    assert (args.run_dir/'latest').resolve()==second
    assert (first/'completed.json').read_text()=='first completion'
    assert (args.run_dir/'completed.json').read_text()=='legacy completion'
    assert (args.run_dir/'manifest.json').read_text()=='legacy manifest'


def test_run_lock_blocks_concurrent_launches_and_accepts_inherited_lease(settings):
    args=arguments()
    with exclusive_run(args.run_dir) as (token,fd):
        assert fd>=0
        with pytest.raises(RuntimeError,match='Another trainer'):
            with exclusive_run(args.run_dir):
                pass
        with pytest.raises(RuntimeError,match='Another trainer'):
            with exclusive_run(args.run_dir,'wrong token'):
                pass
        with exclusive_run(args.run_dir,token):
            pass
        # The joined worker closing its FD must not release the supervisor lock.
        with pytest.raises(RuntimeError):
            with exclusive_run(args.run_dir):
                pass
    with exclusive_run(args.run_dir):
        pass


def test_child_inherited_fd_keeps_run_locked_after_supervisor_closes(settings):
    args=arguments()
    with exclusive_run(args.run_dir) as (_,fd):
        proc=subprocess.Popen([sys.executable,'-c',
            'import sys; print("ready", flush=True); sys.stdin.read()'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,pass_fds=(fd,))
        assert proc.stdout.readline().strip()=='ready'
    try:
        with pytest.raises(RuntimeError,match='Another trainer'):
            with exclusive_run(args.run_dir):
                pass
    finally:
        proc.communicate(input='',timeout=10)
    assert proc.returncode==0
    with exclusive_run(args.run_dir):
        pass


def test_supervisor_records_training_failure_without_using_stale_completion(settings, monkeypatch):
    args=launcher.parse_args(['--run-name','named','--data','unused','--evaluate-final'])
    args.run_dir.mkdir(parents=True)
    (args.run_dir/'completed.json').write_text('{"reason":"stale"}')
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=os.getpid(),wait=lambda:9))
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:pytest.fail('Must not evaluate failed training'))
    assert launcher.launch(args)==9
    log_dir=named_directory(settings,'LOG_DIR',args.run_name)
    assert json.loads((log_dir/'latest'/'training-exit.json').read_text())['returncode']==9


@pytest.mark.parametrize('reason', ['save_and_exit','num_epochs'])
def test_supervisor_appends_logs_and_uses_current_session_completion(settings, monkeypatch, reason):
    args=launcher.parse_args(['--run-name','named','--data','unused','--gpus','0,1','--evaluate-final'])
    args.run_dir.mkdir(parents=True)
    # A stale root completion must never select the wrong checkpoint/reason.
    (args.run_dir/'completed.json').write_text('{"reason":"stale"}')
    log_dir=named_directory(settings,'LOG_DIR',args.run_name)
    log_dir.mkdir(parents=True)
    (log_dir/'training.log').write_text('prior log\n')
    calls=[]
    def popen(command, **kwargs):
        assert kwargs['pass_fds']
        env=kwargs['env']
        with exclusive_run(args.run_dir,env['LLM_JP_LOGKV_RUN_TOKEN']):
            session=create_session(args.run_dir,env['LLM_JP_LOGKV_SESSION_ID'])
            (session/'completed.json').write_text(json.dumps({'reason':reason,'checkpoint':'current-checkpoint'}))
        kwargs['stdout'].write('new training log\n')
        calls.append(command)
        return SimpleNamespace(pid=os.getpid(),wait=lambda:0)
    def evaluate(command, **kwargs):
        assert command[command.index('--adapter')+1]=='current-checkpoint'
        with pytest.raises(RuntimeError,match='Another trainer'):
            with exclusive_run(args.run_dir):
                pass
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess,'Popen',popen)
    monkeypatch.setattr(launcher.subprocess,'run',evaluate)
    assert launcher.launch(args)==0
    assert (log_dir/'training.log').read_text().startswith('prior log\n')
    assert 'new training log' in (log_dir/'training.log').read_text()
    records=log_dir/'latest'
    assert json.loads((records/'training-exit.json').read_text())['returncode']==0
    assert len(calls)==(1 if reason=='save_and_exit' else 2)
    assert (records/('evaluation-skipped.json' if reason=='save_and_exit' else 'evaluation-exit.json')).is_file()
