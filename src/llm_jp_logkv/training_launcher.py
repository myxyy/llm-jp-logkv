"""Named-run torchrun supervisor with append-only logs and launch records."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from . import train
from .training_runs import create_session, exclusive_run, named_directory, new_session_id, resolve_resume


def parse_args(argv=None):
    parser = train.build_parser()
    parser.description = __doc__
    parser.add_argument('--gpus', default='0,1,2,3,4,5')
    parser.add_argument('--evaluate-final', action='store_true')
    parser.add_argument('--evaluation-max-new-tokens', type=int, default=512)
    # Keep accepting the old separator while allowing all options together.
    argv = [value for value in (sys.argv[1:] if argv is None else argv) if value != '--']
    args = train.parse_args(argv, parser)
    gpus = args.gpus.split(',')
    if any(not gpu.isdecimal() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error('--gpus must be a comma-separated list of distinct GPU indices')
    if args.evaluation_max_new_tokens < 1:
        parser.error('--evaluation-max-new-tokens must be positive')
    return args


def trainer_arguments(args):
    result = []
    for action in train.build_parser()._actions:
        if action.dest == 'help':
            continue
        value = getattr(args, action.dest)
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(action.option_strings[0])
        elif value is not None:
            result.extend((action.option_strings[0], str(value)))
    return result


def launch(args):
    with exclusive_run(args.run_dir) as (token, lock_fd):
        args.resume = resolve_resume(args, train.SETTINGS)
        session_id = new_session_id()
        log_dir = named_directory(train.SETTINGS, 'LOG_DIR', args.run_name)
        records = create_session(log_dir, session_id)
        def record(name, value):
            with (records/name).open('x') as stream:
                json.dump(value, stream, indent=2)
                stream.write('\n')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':args.gpus, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
               'TOKENIZERS_PARALLELISM':'false', 'PYTHONUNBUFFERED':'1', 'HF_HUB_DISABLE_PROGRESS_BARS':'1',
               'LLM_JP_LOGKV_RUN_TOKEN':token, 'LLM_JP_LOGKV_SESSION_ID':session_id}
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={len(args.gpus.split(","))}', '-m', 'llm_jp_logkv.train',
                   *trainer_arguments(args)]
        with (log_dir/'training.log').open('a') as log:
            log.write(f'\n=== session {session_id}; resume={args.resume} ===\n')
            log.flush()
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, pass_fds=(lock_fd,))
            record('process.json', {'supervisor_pid':os.getpid(), 'torchrun_pid':proc.pid,
                   'supervisor_start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                   'torchrun_start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().split()[21],
                   'command':command, 'run_dir':str(args.run_dir), 'session_id':session_id,
                   'started_at':datetime.now(timezone.utc).isoformat()})
            print('torchrun PID:', proc.pid, 'run:', args.run_name, 'logs:', log_dir/'training.log', flush=True)
            code = proc.wait()
        record('training-exit.json', {'returncode':code, 'ended_at':datetime.now(timezone.utc).isoformat()})
        if code:
            return code
        if args.evaluate_final:
            completed = json.loads((args.run_dir/'sessions'/session_id/'completed.json').read_text())
            if completed['reason'] == 'save_and_exit':
                record('evaluation-skipped.json', {'reason':'save_and_exit', 'checkpoint':completed['checkpoint']})
                print('Saved and stopped; final evaluation skipped.', flush=True)
            else:
                with (records/'final-evaluation.log').open('x') as log:
                    code = subprocess.run(['llm-jp-logkv', 'evaluate-initial', '--adapter', completed['checkpoint'],
                           '--max-new-tokens', str(args.evaluation_max_new_tokens)],
                           env={**env, 'CUDA_VISIBLE_DEVICES':args.gpus.split(',')[0]},
                           stdout=log, stderr=subprocess.STDOUT).returncode
                record('evaluation-exit.json', {'returncode':code, 'ended_at':datetime.now(timezone.utc).isoformat()})
                if code:
                    return code
        record('supervisor-completed.json', {'completed_at':datetime.now(timezone.utc).isoformat()})
        return 0


def main():
    args = parse_args()
    try:
        code = launch(args)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(code)
