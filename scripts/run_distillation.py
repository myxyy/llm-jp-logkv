#!/usr/bin/env python3
"""Durable supervisor: record torchrun exit status, then evaluate final adapter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from llm_jp_logkv.settings import load_settings
settings=load_settings()
p=argparse.ArgumentParser()
p.add_argument('--gpus',default='0,1,2,3,4,5')
p.add_argument('--log-dir',type=Path,required=True)
p.add_argument('--evaluate-final',action='store_true')
p.add_argument('--evaluation-max-new-tokens',type=int,default=512)
p.add_argument('training_args',nargs=argparse.REMAINDER)
a=p.parse_args()
extra=a.training_args[1:] if a.training_args[:1]==['--'] else a.training_args
if '--run-dir' not in extra:
    p.error('Pass --run-dir to the trainer')
run=Path(extra[extra.index('--run-dir')+1])
if not a.log_dir.resolve().is_relative_to(settings.paths['LOG_DIR'].resolve()):
    p.error('log-dir must be under configured LOG_DIR')
a.log_dir.mkdir(parents=True,exist_ok=False)
def record(name,value):
    with (a.log_dir/name).open('x') as f:
        json.dump(value,f,indent=2)
        f.write('\n')
env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpus,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
     'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1'}
command=[sys.executable,'-m','torch.distributed.run','--standalone',
         f'--nproc_per_node={len(a.gpus.split(","))}','-m','llm_jp_logkv.train',*extra]
with (a.log_dir/'training.log').open('x') as log:
    proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    record('process.json',{'supervisor_pid':os.getpid(),'torchrun_pid':proc.pid,
                          'supervisor_start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                          'torchrun_start_ticks':Path(f'/proc/{proc.pid}/stat').read_text().split()[21],
                          'command':command,'run_dir':str(run),'started_at':datetime.now(timezone.utc).isoformat()})
    print('torchrun PID:',proc.pid,'run:',run,'logs:',a.log_dir,flush=True)
    code=proc.wait()
record('training-exit.json',{'returncode':code,'ended_at':datetime.now(timezone.utc).isoformat()})
if code:
    raise SystemExit(code)
if a.evaluate_final:
    completed=json.loads((run/'completed.json').read_text())
    with (a.log_dir/'final-evaluation.log').open('x') as log:
        code=subprocess.run(['llm-jp-logkv','evaluate-initial','--adapter',completed['checkpoint'],
                             '--max-new-tokens',str(a.evaluation_max_new_tokens)],env={**env,'CUDA_VISIBLE_DEVICES':a.gpus.split(',')[0]},
                            stdout=log,stderr=subprocess.STDOUT).returncode
    record('evaluation-exit.json',{'returncode':code,'ended_at':datetime.now(timezone.utc).isoformat()})
    if code:
        raise SystemExit(code)
record('supervisor-completed.json',{'completed_at':datetime.now(timezone.utc).isoformat()})
