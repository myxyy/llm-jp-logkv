"""Epoch-bounded DDP distillation with optional time limits and immutable checkpoints.

Run through scripts/uv.sh and torchrun. This entry point configures storage
before importing the HF/torch stack.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
from functools import partial
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from uuid import uuid4

from .settings import load_settings
SETTINGS = load_settings()

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer, LlamaForCausalLM

from .distillation import DistillationModel, adapter_state, load_adapter, loss_metrics
from .inference import file_hash, write_json
from .training_data import TokenCache, collate_tokens
from .training_runs import (checkpoint_metadata, completed_checkpoints, create_session, exclusive_run,
                            named_directory, new_session_id, resolve_resume)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run-name',required=True,help='Run name used for checkpoints, logs and experiment records')
    p.add_argument('--resume',help="'latest', a step directory name in this run, or a checkpoint path")
    p.add_argument('--control-file',type=Path,default=Path('control.cmd'),
                   help='Write save_and_exit here to checkpoint and stop at a step boundary')
    p.add_argument('--max-hours',type=float,default=None,
                   help='Optional wall-time limit in hours; omitted means no time limit')
    p.add_argument('--num-epochs',type=int,default=1,
                   help='Total epochs including before resume (default: 1); 0 requires a time or step limit')
    p.add_argument('--max-steps',type=int,default=0,
                   help='Optional total optimizer-step limit; 0 disables it')
    p.add_argument('--sequence-length',type=int,default=1024)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--attention-training',choices=('lora','full'),default='lora',
                   help='Train Q/K/V/O through LoRA (default) or directly; FFN stays frozen')
    p.add_argument('--rank',type=int,default=32,help='LoRA rank (ignored with full attention training)')
    p.add_argument('--alpha',type=float,default=64.,help='LoRA alpha (ignored with full attention training)')
    p.add_argument('--chunk-size',type=int,default=4)
    p.add_argument('--no-rope',action='store_true',help='Disable RoPE in the student only')
    p.add_argument('--conv-kernel-size',type=int,default=0,
                   help='Student causal convolution kernel size (0: disabled; upstream: 4)')
    p.add_argument('--loss-chunk-tokens',type=int,default=512)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--ce-weight',type=float,default=.1)
    p.add_argument('--hidden-mse-weight',type=float,default=0.,
                   help='All-block output MSE weight; mean over layers/features/valid tokens (default: 0, disabled)')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--warmup-steps',type=int,default=50)
    p.add_argument('--sample-every',type=int,default=200)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--max-checkpoints',type=int,default=2,
                   help='Keep the latest N completed checkpoints per run (default: 2; must be >=1)')
    p.add_argument('--validate-every',type=int,default=200)
    p.add_argument('--val-examples-per-rank',type=int,default=8)
    p.add_argument('--sample-max-new-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--skip-initial-samples',action='store_true')
    return p


def parse_args(argv=None, parser=None):
    p=parser or build_parser()
    args=p.parse_args(argv)
    if not math.isfinite(args.hidden_mse_weight) or args.hidden_mse_weight < 0:
        p.error('--hidden-mse-weight must be finite and nonnegative')
    if min(args.sequence_length,args.batch_size,args.grad_accum,args.loss_chunk_tokens,args.val_examples_per_rank,args.sample_max_new_tokens) < 1:
        p.error('Lengths, sizes, and intervals must be positive')
    if args.attention_training=='lora' and args.rank<1:
        p.error('--rank must be positive for LoRA')
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours<=0):
        p.error('--max-hours must be finite and positive; omit it for no time limit')
    if args.num_epochs<0 or args.max_steps<0:
        p.error('--num-epochs and --max-steps must be nonnegative')
    if args.num_epochs==0 and args.max_hours is None and args.max_steps==0:
        p.error('--num-epochs 0 requires --max-hours or --max-steps')
    if args.max_checkpoints<1:
        p.error('--max-checkpoints must be >=1')
    if args.conv_kernel_size<0:
        p.error('--conv-kernel-size must be nonnegative (0 disables convolution)')
    if args.chunk_size < 2 or min(args.sample_every,args.save_every,args.validate_every) < 1:
        p.error('Invalid duration, chunk size, or event interval')
    try:
        args.run_dir=named_directory(SETTINGS,'EXPERIMENT_DIR',args.run_name)
        named_directory(SETTINGS,'CHECKPOINT_DIR',args.run_name)
        named_directory(SETTINGS,'LOG_DIR',args.run_name)
    except ValueError as exc:
        p.error(str(exc))
    return args


def emit(run, name, value):
    with (run/name).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        stream.flush()


def model_config(args):
    lora=args.attention_training=='lora'
    return {'model_id':SETTINGS.model_id,'model_revision':SETTINGS.model_revision,
            'attention_training':args.attention_training,
            'rank':args.rank if lora else None,'alpha':args.alpha if lora else None,'chunk_size':args.chunk_size,
            'no_rope':args.no_rope,'conv_kernel_size':args.conv_kernel_size,
            'no_level_bias':False,'dtype':'bfloat16','lora_parameter_dtype':'float32' if lora else None,
            **({} if lora else {'full_attention_parameter_dtype':'float32'}),
            'targets':['q_proj','k_proj','v_proj','o_proj']}


def check_model_config(saved, args):
    # Pre-option checkpoints used LoRA and RoPE with no convolution.
    saved={'no_rope':False,'conv_kernel_size':0,'attention_training':'lora',**saved}
    if saved!=model_config(args):
        raise ValueError('Resume adapter configuration differs (including training mode, RoPE/convolution)')


def training_signature(args):
    return {key:getattr(args,key) for key in ('sequence_length','batch_size','grad_accum','lr',
        'min_lr_ratio','warmup_steps','temperature','ce_weight','hidden_mse_weight','loss_chunk_tokens','seed')}


def check_resume_signature(saved, args):
    # Old checkpoints included max_hours. Limits may change on resume; model,
    # loss, optimizer hyperparameters and data-order settings must still match.
    saved={key:value for key,value in saved.items() if key not in ('max_hours','num_epochs')}
    saved={'hidden_mse_weight':0.,**saved}
    if saved!=training_signature(args):
        raise ValueError('Resume training/data-order configuration differs')


class EpochBatchIterator:
    """Resume a per-rank data cursor without accumulating across epoch boundaries."""
    def __init__(self, loader, sampler, epoch=0, batch_in_epoch=0):
        self.loader,self.sampler=loader,sampler
        if not len(loader) or epoch<0 or not 0<=batch_in_epoch<=len(loader):
            raise ValueError('Invalid checkpoint epoch/data cursor')
        # Legacy checkpoints can point just past the last batch of an epoch.
        self.epoch=epoch+batch_in_epoch//len(loader)
        self.batch_in_epoch=batch_in_epoch%len(loader)
        self.iterator=None

    def next_update(self, grad_accum):
        if grad_accum<1:
            raise ValueError('grad_accum must be positive')
        if self.iterator is None:
            self.sampler.set_epoch(self.epoch)
            self.iterator=iter(self.loader)
            for _ in range(self.batch_in_epoch):
                next(self.iterator)
        count=min(grad_accum,len(self.loader)-self.batch_in_epoch)
        batches=[next(self.iterator) for _ in range(count)]
        self.batch_in_epoch+=count
        if self.batch_in_epoch==len(self.loader):
            self.epoch+=1
            self.batch_in_epoch=0
            self.iterator=None
        return batches


STOP_REASONS = (None, 'max_steps', 'deadline', 'stop_request', 'save_and_exit', 'num_epochs')


def consume_save_and_exit(path):
    """Only rank zero calls this; leave empty/unknown commands untouched."""
    try:
        if path.read_text(encoding='utf-8').strip() != 'save_and_exit':
            return False
        path.unlink(missing_ok=True)
    except FileNotFoundError:
        return False
    return True


def training_control(args, step, elapsed, stop_requested, device, *, epoch, batch_in_epoch, batches_per_epoch,
                     group=None):
    """Broadcast the same stop reason and LR to every rank before the next update."""
    control=torch.zeros(3,device=device,dtype=torch.float64)
    if dist.get_rank()==0:
        # An explicit save-and-exit takes priority, including at the deadline.
        if consume_save_and_exit(args.control_file):
            reason='save_and_exit'
            print(f'save_and_exit received from {args.control_file} at step {step}',flush=True)
        elif args.max_steps and step>=args.max_steps:
            reason='max_steps'
        elif args.num_epochs and epoch>=args.num_epochs:
            reason='num_epochs'
        elif args.max_hours is not None and elapsed>=args.max_hours*3600:
            reason='deadline'
        elif stop_requested or (args.run_dir/'STOP').exists():
            reason='stop_request'
        else:
            reason=None
        if args.num_epochs:
            progress=(epoch+batch_in_epoch/batches_per_epoch)/args.num_epochs
        elif args.max_hours is not None:
            progress=elapsed/(args.max_hours*3600)
        else:
            progress=step/args.max_steps
        progress=min(progress,1.)
        lr=args.lr*min(1.,(step+1)/max(args.warmup_steps,1))*(args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*progress)))
        control[:]=torch.tensor([elapsed,STOP_REASONS.index(reason),lr],device=device)
    if group is None:
        dist.broadcast(control,src=0)
    else:
        dist.broadcast(control,src=0,group=group)
    elapsed,reason_code,lr=control.tolist()
    return elapsed,STOP_REASONS[int(reason_code)],lr


def prune_checkpoints(root, run_dir, keep):
    """Rank zero prunes only complete checkpoints owned by this run, by step."""
    if keep<1:
        raise ValueError('Checkpoint retention must be >=1')
    for path in completed_checkpoints(root,run_dir)[:-keep]:
        step=int(path.name.removeprefix('step-'))
        shutil.rmtree(path)
        emit(run_dir,'checkpoint-pruning.jsonl',{'step':step,'path':str(path),
             'max_checkpoints':keep,'deleted_at':datetime.now(timezone.utc).isoformat()})
        print(f'Pruned checkpoint step={step} path={path}',flush=True)


def save_checkpoint(model, optimizer, args, step, epoch, batch_in_epoch, elapsed, world, rank):
    local_rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
    rngs=[None]*world if rank==0 else None
    dist.gather_object(local_rng,rngs,dst=0)
    root=SETTINGS.paths['CHECKPOINT_DIR']/args.run_dir.name
    path=root/f'step-{step:07d}'
    if rank==0:
        if path.exists():
            if checkpoint_metadata(path) is None:
                # Preserve an interrupted legacy save under a name excluded from
                # latest/retention, allowing the same step to be saved again.
                if path.is_symlink() or not path.is_dir():
                    raise RuntimeError(f'Unexpected checkpoint path: {path}')
                path.rename(root/f'{path.name}.incomplete-{uuid4().hex[:12]}')
        if path.exists():
            complete=json.loads((path/'complete.json').read_text())
            if complete['step']!=step or Path(complete['source_run']).resolve()!=args.run_dir.resolve():
                raise RuntimeError(f'Existing checkpoint belongs to another run or step: {path}')
        else:
            temporary=root/f'.{path.name}-{uuid4().hex[:12]}.tmp'
            temporary.mkdir(parents=True,exist_ok=False)
            torch.save(adapter_state(model.student),temporary/'adapter.pt')
            torch.save({'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,
                        'batch_in_epoch':batch_in_epoch,'elapsed_seconds':elapsed,
                        'rngs':rngs,'world_size':world,'training_signature':training_signature(args),
                        'run_start_unix':args.run_start_unix},temporary/'training_state.pt')
            write_json(temporary/'config.json',model_config(args))
            write_json(temporary/'complete.json',{'step':step,'elapsed_seconds':elapsed,
                       'adapter_sha256':file_hash(temporary/'adapter.pt'),
                       'training_state_sha256':file_hash(temporary/'training_state.pt'),
                       'source_run':str(args.run_dir),'data_manifest_sha256':file_hash(args.data/'manifest.json')})
            temporary.rename(path)
        emit(args.run_dir,'checkpoints.jsonl',{'step':step,'path':str(path),'elapsed_seconds':elapsed})
        print(f'checkpoint step={step} path={path}',flush=True)
        # Never prune before all checkpoint files and the completion marker are saved.
        prune_checkpoints(root,args.run_dir,args.max_checkpoints)
    dist.barrier()
    return path


@torch.no_grad()
def validate(model, loader, device):
    totals=torch.zeros(model.stats_size,device=device,dtype=torch.float64)
    for ids,lengths in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,stats=model(ids.to(device,non_blocking=True),lengths.to(device,non_blocking=True))
        if not torch.isfinite(stats).all().item():
            raise FloatingPointError('Non-finite validation metrics')
        totals+=stats.double()
    dist.all_reduce(totals)
    return {**loss_metrics(totals,model.ce_weight,model.hidden_mse_weight),
            'tokens':int(totals[2].item())}


def sample_prompts():
    suite=json.loads(Path('evaluation/initial-quality.json').read_text())
    return suite['prompts']+[
        {'id':'translation','prompt':'次の英文を日本語に訳してください。The library is closed on Mondays.'},
        {'id':'ordering','prompt':'次の数を小さい順に並べてください。9、2、15、4。答えだけを書いてください。'}]


@torch.inference_mode()
def record_samples(model, tokenizer, args, step, rank, world, device, phase='periodic'):
    results=[]
    for index,item in enumerate(sample_prompts()):
        if index % world != rank:
            continue
        ids=tokenizer.apply_chat_template([{'role':'user','content':item['prompt']}],
            tokenize=True,add_generation_prompt=True,reasoning_effort='low',
            conversation_start_date='2026-11-02',return_tensors='pt',return_dict=True).input_ids.to(device)
        start=time.perf_counter()
        generated,state,reason=model.student.greedy_generate(ids,max_new_tokens=args.sample_max_new_tokens,
            eos_token_id=model.student.config.eos_token_id,prefill_chunk_size=128)
        results.append({'id':item['id'],'prompt':item['prompt'],'rank':rank,
                        'input_tokens':ids.shape[1],'output_tokens':generated.shape[1],
                        'input_token_ids':ids[0].tolist(),'token_ids':generated[0].tolist(),
                        'text':tokenizer.decode(generated[0],skip_special_tokens=False),
                        'stop_reason':reason,'seconds':time.perf_counter()-start,
                        'state_tensor_bytes':state.tensor_bytes})
    collected=[None]*world if rank==0 else None
    dist.gather_object(results,collected,dst=0)
    if rank==0:
        path=args.session_dir/'samples'/f'{phase}-step-{step:07d}.json'
        with path.open('x',encoding='utf-8') as output:
            json.dump({'step':step,'phase':phase,'created_at':datetime.now(timezone.utc).isoformat(),
                       'samples':[r for group in collected for r in group]},output,indent=2,ensure_ascii=False)
            output.write('\n')
        print(f'samples step={step} path={path}',flush=True)
    dist.barrier()


def main():
    args=parse_args()
    local_rank=int(os.environ['LOCAL_RANK'])
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    try:
        with ExitStack() as stack:
            launch=[None]
            if rank==0:
                try:
                    stack.enter_context(exclusive_run(args.run_dir,os.environ.get('LLM_JP_LOGKV_RUN_TOKEN')))
                    resume=resolve_resume(args,SETTINGS)
                    session_id=os.environ.get('LLM_JP_LOGKV_SESSION_ID') or new_session_id()
                    create_session(args.run_dir,session_id)
                    launch[0]={'resume':str(resume) if resume else None,'session_id':session_id}
                except (OSError,ValueError,RuntimeError) as exc:
                    launch[0]={'error':str(exc)}
            dist.broadcast_object_list(launch,src=0)
            if 'error' in launch[0]:
                raise RuntimeError(launch[0]['error'])
            args.resume=Path(launch[0]['resume']) if launch[0]['resume'] else None
            args.session_id=launch[0]['session_id']
            args.session_dir=args.run_dir/'sessions'/args.session_id
            run_training(args,device,rank,world)
    finally:
        dist.destroy_process_group()


def run_training(args, device, rank, world):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(args.seed)
    stop_requested={'value':False}
    def request_stop(signum,frame):
        stop_requested['value']=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    if rank==0:
        (args.session_dir/'samples').mkdir()
        data_manifest=json.loads((args.data/'manifest.json').read_text())
        if data_manifest['tokenizer_revision']!=SETTINGS.model_revision:
            raise ValueError('Token cache was built for a different base revision')
        for name,digest in data_manifest['files'].items():
            if file_hash(args.data/name)!=digest:
                raise ValueError(f'Token cache checksum mismatch: {name}')
        root=Path.cwd()
        write_json(args.session_dir/'manifest.json',{
            'created_at':datetime.now(timezone.utc).isoformat(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            'source_files_sha256':{str(p):file_hash(p) for p in sorted(Path('src').rglob('*.py'))},
            'uv_lock_sha256':file_hash(root/'uv.lock'),
            'model':model_config(args),'data_manifest':data_manifest,
            'data_manifest_sha256':file_hash(args.data/'manifest.json'),
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'packages':{n:version(n) for n in ('torch','transformers','huggingface-hub','accelerate')},
            'world_size':world,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu':torch.cuda.get_device_name(device),'teacher_backend':'sdpa',
            'optimizer':{'name':'AdamW','fused':True,'betas':[.9,.95],
                         'weight_decay':.01,'gradient_clip_norm':1.},
            'sampling':{'reasoning_effort':'low','conversation_start_date':'2026-11-02',
                        'generation':'greedy','prefill_chunk_size':128},
            'objective':'Exact full-vocabulary KL(teacher || student) * T^2 + ce_weight * next-token CE + hidden_mse_weight * mean all-block MSE; all nonpadding targets',
            'checkpointing':'non-reentrant decoder-layer and vocabulary-loss checkpoints; grouped LogKV',
            'time_budget':'unlimited unless max_hours is explicit; then wall time from original training start including downtime and periodic work',
            'lr_schedule':'epoch progress' if args.num_epochs else 'wall time' if args.max_hours is not None else 'optimizer steps',
            'epoch_policy':'per-rank loader pass; sampler and batch drop_last; partial accumulation at epoch end, no crossing into next epoch'})
    dist.barrier()
    tokenizer=AutoTokenizer.from_pretrained(SETTINGS.model_path,local_files_only=True)
    teacher,loading=LlamaForCausalLM.from_pretrained(SETTINGS.model_path,local_files_only=True,
        dtype=torch.bfloat16,device_map={'':str(device)},attn_implementation='sdpa',output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Base model weights were not loaded exactly')
    torch.manual_seed(args.seed)
    model=DistillationModel(teacher,rank=args.rank,alpha=args.alpha,chunk_size=args.chunk_size,
        attention_training=args.attention_training,
        no_rope=args.no_rope,conv_kernel_size=args.conv_kernel_size,
        loss_chunk_tokens=args.loss_chunk_tokens,temperature=args.temperature,ce_weight=args.ce_weight,
        hidden_mse_weight=args.hidden_mse_weight)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.95),weight_decay=.01,fused=True)
    step=epoch=batch_in_epoch=0; prior_elapsed=0.
    if args.resume:
        config=json.loads((args.resume/'config.json').read_text())
        check_model_config(config,args)
        complete=json.loads((args.resume/'complete.json').read_text())
        if complete['data_manifest_sha256']!=file_hash(args.data/'manifest.json'):
            raise ValueError('Resume data differs')
        if complete['adapter_sha256']!=file_hash(args.resume/'adapter.pt'):
            raise ValueError('Adapter checksum mismatch')
        load_adapter(model.student,torch.load(args.resume/'adapter.pt',map_location='cpu',weights_only=True))
        if complete['training_state_sha256']!=file_hash(args.resume/'training_state.pt'):
            raise ValueError('Training state checksum mismatch')
        saved=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=True)
        check_resume_signature(saved['training_signature'],args)
        if saved['world_size']!=world:
            raise ValueError('Resume requires the original world size')
        optimizer.load_state_dict(saved['optimizer'])
        step,epoch,batch_in_epoch=saved['step'],saved['epoch'],saved['batch_in_epoch']
        args.run_start_unix=saved['run_start_unix']
        prior_elapsed=time.time()-args.run_start_unix
        torch.set_rng_state(saved['rngs'][rank]['cpu'])
        torch.cuda.set_rng_state(saved['rngs'][rank]['cuda'],device)
        del saved
    ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=False,
            gradient_as_bucket_view=True)
    train_data=TokenCache(args.data,'train',args.sequence_length)
    valid_data=TokenCache(args.data,'valid',args.sequence_length)
    sampler=DistributedSampler(train_data,num_replicas=world,rank=rank,shuffle=True,seed=args.seed,drop_last=True)
    collate=partial(collate_tokens,sequence_length=args.sequence_length,pad_token_id=tokenizer.pad_token_id)
    loader=DataLoader(train_data,batch_size=args.batch_size,sampler=sampler,drop_last=True,
                      collate_fn=collate,pin_memory=True,num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))
    if not len(loader):
        raise ValueError('Too few training examples for global batch')
    cursor=EpochBatchIterator(loader,sampler,epoch,batch_in_epoch)
    val_order=np.random.default_rng(args.seed+1).permutation(len(valid_data)).tolist()
    val_indices=val_order[rank::world]
    val_loader=DataLoader(Subset(valid_data,val_indices[:args.val_examples_per_rank]),
                         batch_size=args.batch_size,collate_fn=collate,pin_memory=True)
    final_val_loader=DataLoader(Subset(valid_data,val_indices),batch_size=args.batch_size,
                               collate_fn=collate,pin_memory=True)
    if rank==0:
        write_json(args.session_dir/'runtime.json',{'teacher_parameters':sum(p.numel() for p in teacher.parameters()),
            'trainable_parameters':sum(p.numel() for p in params),'layers':len(model.student.layers),
            'train_examples':len(train_data),'valid_examples':len(valid_data),
            'global_examples_per_step':world*args.batch_size*args.grad_accum,
            'batches_per_epoch_per_rank':len(loader),'optimizer_steps_per_epoch':math.ceil(len(loader)/args.grad_accum),
            'num_epochs':args.num_epochs,'max_hours':args.max_hours,
            'grouped_forward':True,'lora_only':args.attention_training=='lora' and not bool(args.conv_kernel_size),
            'attention_training':args.attention_training,'trainable_attention_parameter_dtype':'float32',
            'conv_kernel_size':args.conv_kernel_size,'no_rope':args.no_rope,
            'conv_parameter_dtype':'float32' if args.conv_kernel_size else None,
            'frozen_teacher':True})
    dist.barrier()
    started=time.monotonic()
    if not args.resume:
        args.run_start_unix=time.time()
    else:
        prior_elapsed=time.time()-args.run_start_unix
    if rank==0:
        remaining_seconds=None if args.max_hours is None else max(0,args.max_hours*3600-prior_elapsed)
        write_json(args.session_dir/'started.json',{'started_at':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(),'remaining_seconds':remaining_seconds,'step':step,
            'num_epochs':args.num_epochs,'epoch':cursor.epoch,'batch_in_epoch':cursor.batch_in_epoch})
        print(f'Training limits: epochs={args.num_epochs or "unlimited"}, '
              f'time={str(args.max_hours)+" hours" if args.max_hours is not None else "unlimited"}, '
              f'steps={args.max_steps or "unlimited"}; batches per epoch/rank={len(loader)}',flush=True)
        print(f'Checkpoint retention: latest {args.max_checkpoints} completed checkpoints per run',flush=True)
    initial_validation=validate(model,val_loader,device)
    if rank==0:
        emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'initial',**initial_validation})
    initial_all=validate(model,final_val_loader,device)
    if rank==0:
        emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'initial_all_heldout',**initial_all})
    if not args.skip_initial_samples:
        record_samples(model,tokenizer,args,step,rank,world,device,phase='initial')
    while True:
        elapsed,reason,lr=training_control(args,step,prior_elapsed+time.monotonic()-started,
                                          stop_requested['value'],device,epoch=cursor.epoch,
                                          batch_in_epoch=cursor.batch_in_epoch,batches_per_epoch=len(loader))
        if reason is not None:
            break
        for group in optimizer.param_groups:
            group['lr']=lr
        batches=cursor.next_update(args.grad_accum)
        count=torch.tensor(sum(int((lengths-1).sum()) for _,lengths in batches),device=device,dtype=torch.float64)
        dist.all_reduce(count)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device);tick=time.perf_counter()
        totals=torch.zeros(model.stats_size,device=device,dtype=torch.float64)
        from contextlib import nullcontext
        for micro,(ids,lengths) in enumerate(batches):
            context=ddp.no_sync() if micro+1<len(batches) else nullcontext()
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,stats=ddp(ids.to(device,non_blocking=True),lengths.to(device,non_blocking=True))
                    scaled=loss*(world/count)
                finite=torch.tensor(int(torch.isfinite(loss).item()),device=device)
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Non-finite distributed loss')
                scaled.backward()
                totals+=stats.double()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step()
        dist.all_reduce(totals)
        torch.cuda.synchronize(device)
        step+=1
        seconds=time.perf_counter()-tick
        peak=torch.tensor(torch.cuda.max_memory_allocated(device),device=device,dtype=torch.float64)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if rank==0:
            metrics={'step':step,'elapsed_seconds':prior_elapsed+time.monotonic()-started,
                'lr':lr,**loss_metrics(totals,args.ce_weight,args.hidden_mse_weight),
                'grad_norm':norm.item(),'valid_tokens':int(totals[2].item()),'step_seconds':seconds,
                'epoch':cursor.epoch,'batch_in_epoch':cursor.batch_in_epoch,'microbatches_per_rank':len(batches),
                'valid_tokens_per_second':totals[2].item()/seconds,'max_rank_peak_allocated_bytes':int(peak.item())}
            emit(args.run_dir,'metrics.jsonl',metrics)
            if step==1 or step%10==0:
                print(json.dumps(metrics),flush=True)
            if step==1 or step%20==0:
                telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                emit(args.run_dir,'gpu.jsonl',{'step':step,'read_at':datetime.now(timezone.utc).isoformat(),'csv':telemetry})
        if step%args.save_every==0:
            save_checkpoint(model,optimizer,args,step,cursor.epoch,cursor.batch_in_epoch,prior_elapsed+time.monotonic()-started,world,rank)
        if step%args.validate_every==0:
            values=validate(model,val_loader,device)
            if rank==0:
                emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'periodic',**values})
        if step%args.sample_every==0:
            record_samples(model,tokenizer,args,step,rank,world,device)
    # Verify every data-parallel replica reached identical adapter weights.
    import hashlib
    digest=hashlib.sha256()
    for name, tensor in adapter_state(model.student).items():
        digest.update(name.encode());digest.update(tensor.numpy().tobytes())
    hashes=[None]*world
    dist.all_gather_object(hashes,digest.hexdigest())
    if len(set(hashes))!=1:
        raise RuntimeError('Distributed adapter replicas diverged')
    final_path=save_checkpoint(model,optimizer,args,step,cursor.epoch,cursor.batch_in_epoch,prior_elapsed+time.monotonic()-started,world,rank)
    values=None
    if reason!='save_and_exit':
        values=validate(model,final_val_loader,device)
        if rank==0:
            emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'final_all_heldout',**values})
        record_samples(model,tokenizer,args,step,rank,world,device,phase='final')
    if rank==0:
        write_json(args.session_dir/'completed.json',{'step':step,'reason':reason,'elapsed_seconds':prior_elapsed+time.monotonic()-started,
            'epoch':cursor.epoch,'batch_in_epoch':cursor.batch_in_epoch,
            'completed_at':datetime.now(timezone.utc).isoformat(),'checkpoint':str(final_path),'validation':values,'adapter_hashes_by_rank':hashes})
        print(f'Training finished: {reason}; checkpoint={final_path}',flush=True)


if __name__=='__main__':
    main()
