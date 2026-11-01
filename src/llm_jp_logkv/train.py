"""Time-bounded DDP distillation with immutable checkpoints and periodic samples.

Run through scripts/uv.sh and torchrun. This entry point configures storage
before importing the HF/torch stack.
"""
import argparse
from datetime import datetime, timezone, timedelta
from functools import partial
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .settings import load_settings
SETTINGS = load_settings()

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer, LlamaForCausalLM

from .distillation import DistillationModel, adapter_state, load_adapter
from .inference import file_hash, write_json
from .training_data import TokenCache, collate_tokens


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--max-hours',type=float,default=8.)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--sequence-length',type=int,default=1024)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--rank',type=int,default=32)
    p.add_argument('--alpha',type=float,default=64.)
    p.add_argument('--chunk-size',type=int,default=4)
    p.add_argument('--loss-chunk-tokens',type=int,default=512)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--ce-weight',type=float,default=.1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--warmup-steps',type=int,default=50)
    p.add_argument('--sample-every',type=int,default=200)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--validate-every',type=int,default=200)
    p.add_argument('--val-examples-per-rank',type=int,default=8)
    p.add_argument('--sample-max-new-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--skip-initial-samples',action='store_true')
    args=p.parse_args()
    if min(args.sequence_length,args.batch_size,args.grad_accum,args.rank,args.loss_chunk_tokens,args.val_examples_per_rank,args.sample_max_new_tokens) < 1:
        p.error('Lengths, sizes, and intervals must be positive')
    if args.max_hours <= 0 or args.chunk_size < 2 or min(args.sample_every,args.save_every,args.validate_every) < 1:
        p.error('Invalid duration, chunk size, or event interval')
    if not args.run_dir.resolve().is_relative_to(SETTINGS.paths['EXPERIMENT_DIR'].resolve()):
        p.error('run-dir must be under configured EXPERIMENT_DIR')
    return args


def emit(run, name, value):
    with (run/name).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        stream.flush()


def model_config(args):
    return {'model_id':SETTINGS.model_id,'model_revision':SETTINGS.model_revision,
            'rank':args.rank,'alpha':args.alpha,'chunk_size':args.chunk_size,
            'no_level_bias':False,'dtype':'bfloat16','lora_parameter_dtype':'float32',
            'targets':['q_proj','k_proj','v_proj','o_proj']}


def training_signature(args):
    return {key:getattr(args,key) for key in ('sequence_length','batch_size','grad_accum','lr',
        'min_lr_ratio','warmup_steps','temperature','ce_weight','loss_chunk_tokens','seed','max_hours')}


def save_checkpoint(model, optimizer, args, step, epoch, batch_in_epoch, elapsed, world, rank):
    local_rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
    rngs=[None]*world if rank==0 else None
    dist.gather_object(local_rng,rngs,dst=0)
    root=SETTINGS.paths['CHECKPOINT_DIR']/args.run_dir.name
    path=root/f'step-{step:07d}'
    if rank==0:
        if path.exists():
            if not (path/'complete.json').is_file():
                raise RuntimeError(f'Incomplete existing checkpoint: {path}')
        else:
            path.mkdir(parents=True,exist_ok=False)
            torch.save(adapter_state(model.student),path/'adapter.pt')
            torch.save({'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,
                        'batch_in_epoch':batch_in_epoch,'elapsed_seconds':elapsed,
                        'rngs':rngs,'world_size':world,'training_signature':training_signature(args),
                        'run_start_unix':args.run_start_unix},path/'training_state.pt')
            write_json(path/'config.json',model_config(args))
            write_json(path/'complete.json',{'step':step,'elapsed_seconds':elapsed,
                       'adapter_sha256':file_hash(path/'adapter.pt'),
                       'training_state_sha256':file_hash(path/'training_state.pt'),
                       'source_run':str(args.run_dir),'data_manifest_sha256':file_hash(args.data/'manifest.json')})
        emit(args.run_dir,'checkpoints.jsonl',{'step':step,'path':str(path),'elapsed_seconds':elapsed})
        print(f'checkpoint step={step} path={path}',flush=True)
    dist.barrier()
    return path


@torch.no_grad()
def validate(model, loader, device):
    totals=torch.zeros(3,device=device,dtype=torch.float64)
    for ids,lengths in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,stats=model(ids.to(device,non_blocking=True),lengths.to(device,non_blocking=True))
        if not torch.isfinite(stats).all().item():
            raise FloatingPointError('Non-finite validation metrics')
        totals+=stats.double()
    dist.all_reduce(totals)
    return {'kl':(totals[0]/totals[2]).item(),'ce':(totals[1]/totals[2]).item(),
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
        path=args.run_dir/'samples'/f'{phase}-step-{step:07d}.json'
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
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(args.seed)
    stop_requested={'value':False}
    def request_stop(signum,frame):
        stop_requested['value']=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    if rank==0:
        args.run_dir.mkdir(parents=True,exist_ok=False)
        (args.run_dir/'samples').mkdir()
        data_manifest=json.loads((args.data/'manifest.json').read_text())
        if data_manifest['tokenizer_revision']!=SETTINGS.model_revision:
            raise ValueError('Token cache was built for a different base revision')
        for name,digest in data_manifest['files'].items():
            if file_hash(args.data/name)!=digest:
                raise ValueError(f'Token cache checksum mismatch: {name}')
        root=Path.cwd()
        write_json(args.run_dir/'manifest.json',{
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
            'objective':'Exact full-vocabulary KL(teacher || student) * T^2 + ce_weight * next-token CE; all nonpadding targets',
            'checkpointing':'non-reentrant decoder-layer and vocabulary-loss checkpoints; grouped LogKV',
            'time_budget':'wall time from training start including periodic sampling/validation/saving; finalization may add a short tail',
            'authorization':'User authorized an eight-hour trial and all six GPUs for this run.'})
    dist.barrier()
    tokenizer=AutoTokenizer.from_pretrained(SETTINGS.model_path,local_files_only=True)
    teacher,loading=LlamaForCausalLM.from_pretrained(SETTINGS.model_path,local_files_only=True,
        dtype=torch.bfloat16,device_map={'':str(device)},attn_implementation='sdpa',output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Base model weights were not loaded exactly')
    torch.manual_seed(args.seed)
    model=DistillationModel(teacher,rank=args.rank,alpha=args.alpha,chunk_size=args.chunk_size,
        loss_chunk_tokens=args.loss_chunk_tokens,temperature=args.temperature,ce_weight=args.ce_weight)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.95),weight_decay=.01,fused=True)
    step=epoch=batch_in_epoch=0; prior_elapsed=0.
    if args.resume:
        config=json.loads((args.resume/'config.json').read_text())
        if config!=model_config(args):
            raise ValueError('Resume adapter configuration differs')
        complete=json.loads((args.resume/'complete.json').read_text())
        if complete['data_manifest_sha256']!=file_hash(args.data/'manifest.json'):
            raise ValueError('Resume data differs')
        if complete['adapter_sha256']!=file_hash(args.resume/'adapter.pt'):
            raise ValueError('Adapter checksum mismatch')
        load_adapter(model.student,torch.load(args.resume/'adapter.pt',map_location='cpu',weights_only=True))
        if complete['training_state_sha256']!=file_hash(args.resume/'training_state.pt'):
            raise ValueError('Training state checksum mismatch')
        saved=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=True)
        if saved['training_signature']!=training_signature(args):
            raise ValueError('Resume training/data-order configuration differs')
        if saved['world_size']!=world:
            raise ValueError('Resume requires the original world size')
        optimizer.load_state_dict(saved['optimizer'])
        step,epoch,batch_in_epoch=saved['step'],saved['epoch'],saved['batch_in_epoch']
        args.run_start_unix=saved['run_start_unix']
        prior_elapsed=time.time()-args.run_start_unix
        torch.set_rng_state(saved['rngs'][rank]['cpu'])
        torch.cuda.set_rng_state(saved['rngs'][rank]['cuda'],device)
        del saved
    ddp=DDP(model,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False,
            gradient_as_bucket_view=True)
    train_data=TokenCache(args.data,'train',args.sequence_length)
    valid_data=TokenCache(args.data,'valid',args.sequence_length)
    sampler=DistributedSampler(train_data,num_replicas=world,rank=rank,shuffle=True,seed=args.seed,drop_last=True)
    collate=partial(collate_tokens,sequence_length=args.sequence_length,pad_token_id=tokenizer.pad_token_id)
    loader=DataLoader(train_data,batch_size=args.batch_size,sampler=sampler,drop_last=True,
                      collate_fn=collate,pin_memory=True,num_workers=0)
    if not len(loader):
        raise ValueError('Too few training examples for global batch')
    val_order=np.random.default_rng(args.seed+1).permutation(len(valid_data)).tolist()
    val_indices=val_order[rank::world]
    val_loader=DataLoader(Subset(valid_data,val_indices[:args.val_examples_per_rank]),
                         batch_size=args.batch_size,collate_fn=collate,pin_memory=True)
    final_val_loader=DataLoader(Subset(valid_data,val_indices),batch_size=args.batch_size,
                               collate_fn=collate,pin_memory=True)
    if rank==0:
        write_json(args.run_dir/'runtime.json',{'teacher_parameters':sum(p.numel() for p in teacher.parameters()),
            'trainable_parameters':sum(p.numel() for p in params),'layers':len(model.student.layers),
            'train_examples':len(train_data),'valid_examples':len(valid_data),
            'global_examples_per_step':world*args.batch_size*args.grad_accum,
            'grouped_forward':True,'lora_only':True,'frozen_teacher':True})
    sampler.set_epoch(epoch); iterator=iter(loader)
    for _ in range(batch_in_epoch):
        next(iterator)
    def next_batch():
        nonlocal epoch,batch_in_epoch,iterator
        try:
            batch=next(iterator)
        except StopIteration:
            epoch+=1;batch_in_epoch=0;sampler.set_epoch(epoch);iterator=iter(loader)
            batch=next(iterator)
        batch_in_epoch+=1
        return batch
    dist.barrier()
    started=time.monotonic()
    if not args.resume:
        args.run_start_unix=time.time()
    else:
        prior_elapsed=time.time()-args.run_start_unix
    if rank==0:
        write_json(args.run_dir/'started.json',{'started_at':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(),'remaining_seconds':max(0,args.max_hours*3600-prior_elapsed),'step':step})
    initial_validation=validate(model,val_loader,device)
    if rank==0:
        emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'initial',**initial_validation})
    initial_all=validate(model,final_val_loader,device)
    if rank==0:
        emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'initial_all_heldout',**initial_all})
    if not args.skip_initial_samples:
        record_samples(model,tokenizer,args,step,rank,world,device,phase='initial')
    reason='deadline'
    while True:
        control=torch.zeros(3,device=device,dtype=torch.float64)
        if rank==0:
            elapsed=prior_elapsed+time.monotonic()-started
            stop=elapsed>=args.max_hours*3600 or (args.max_steps and step>=args.max_steps) or stop_requested['value'] or (args.run_dir/'STOP').exists()
            progress=min(elapsed/(args.max_hours*3600),1.)
            lr=args.lr*min(1.,(step+1)/max(args.warmup_steps,1))*(args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*progress)))
            control[:]=torch.tensor([elapsed,float(bool(stop)),lr],device=device)
        dist.broadcast(control,src=0)
        elapsed,stop,lr=control.tolist()
        if stop:
            reason=('max_steps' if args.max_steps and step>=args.max_steps else
                    'deadline' if elapsed>=args.max_hours*3600 else 'stop_request')
            break
        for group in optimizer.param_groups:
            group['lr']=lr
        batches=[next_batch() for _ in range(args.grad_accum)]
        count=torch.tensor(sum(int((lengths-1).sum()) for _,lengths in batches),device=device,dtype=torch.float64)
        dist.all_reduce(count)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device);tick=time.perf_counter()
        totals=torch.zeros(3,device=device,dtype=torch.float64)
        from contextlib import nullcontext
        for micro,(ids,lengths) in enumerate(batches):
            context=ddp.no_sync() if micro+1<args.grad_accum else nullcontext()
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
                'lr':lr,'kl':(totals[0]/totals[2]).item(),'ce':(totals[1]/totals[2]).item(),
                'loss':((totals[0]+args.ce_weight*totals[1])/totals[2]).item(),
                'grad_norm':norm.item(),'valid_tokens':int(totals[2].item()),'step_seconds':seconds,
                'valid_tokens_per_second':totals[2].item()/seconds,'max_rank_peak_allocated_bytes':int(peak.item())}
            emit(args.run_dir,'metrics.jsonl',metrics)
            if step==1 or step%10==0:
                print(json.dumps(metrics),flush=True)
            if step==1 or step%20==0:
                telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                emit(args.run_dir,'gpu.jsonl',{'step':step,'read_at':datetime.now(timezone.utc).isoformat(),'csv':telemetry})
        if step%args.save_every==0:
            save_checkpoint(model,optimizer,args,step,epoch,batch_in_epoch,prior_elapsed+time.monotonic()-started,world,rank)
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
    final_path=save_checkpoint(model,optimizer,args,step,epoch,batch_in_epoch,prior_elapsed+time.monotonic()-started,world,rank)
    values=validate(model,final_val_loader,device)
    if rank==0:
        emit(args.run_dir,'validation.jsonl',{'step':step,'phase':'final_all_heldout',**values})
    record_samples(model,tokenizer,args,step,rank,world,device,phase='final')
    if rank==0:
        write_json(args.run_dir/'completed.json',{'step':step,'reason':reason,'elapsed_seconds':prior_elapsed+time.monotonic()-started,
            'completed_at':datetime.now(timezone.utc).isoformat(),'checkpoint':str(final_path),'validation':values,'adapter_hashes_by_rank':hashes})
        print('Training completed',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
