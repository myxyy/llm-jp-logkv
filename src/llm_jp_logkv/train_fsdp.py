"""Dedicated teacher + student-only FSDP2 full-attention distillation.

Use scripts/run_distillation_fsdp.py. The last torchrun rank is the teacher.
Frozen student modules are replicated; attention and optional conv are sharded.
"""
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import uuid4

from . import train
SETTINGS = train.SETTINGS  # Settings are loaded before torch/HF imports.

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, LlamaForCausalLM

from .distillation import load_adapter, loss_metrics
from .inference import file_hash, write_json
from .remote_fsdp import (RemoteStudent, TeacherChannel, shard_student, clip_sharded_grad_norm,
                          full_adapter_state, local_optimizer_state, restore_local_optimizer)
from .training_data import TokenCache, collate_tokens
from .training_runs import (checkpoint_metadata, create_session, exclusive_run,
                            new_session_id, resolve_resume)

BACKEND = 'remote-teacher-fsdp2-attention-v1'


def log_progress(args, rank, event, **fields):
    if rank == 0:
        record = {'event': event, 'time': datetime.now(timezone.utc).isoformat(), **fields}
        train.emit(args.session_dir, 'progress.jsonl', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)


def build_parser():
    parser = train.build_parser()
    parser.description = __doc__
    parser.set_defaults(attention_training='full', loss_chunk_tokens=64)
    for action in parser._actions:
        if action.dest == 'attention_training':
            action.choices = ('full',)
            action.help = 'Full Q/K/V/O tuning (default); FFN stays frozen'
    parser.add_argument('--skip-samples', action='store_true', help='Skip all text generation samples')
    parser.add_argument('--skip-final-validation', action='store_true',
                        help='Skip full held-out validation on exit (useful for VRAM preflight)')
    return parser


def parse_args(argv=None, parser=None):
    parser = parser or build_parser()
    args = train.parse_args(argv, parser)
    if args.attention_training != 'full':
        parser.error('This entry point requires --attention-training full')
    return args


def load_source(device):
    model, loading = LlamaForCausalLM.from_pretrained(
        SETTINGS.model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': str(device)}, attn_implementation='sdpa', output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('Base model weights were not loaded exactly')
    return model.requires_grad_(False).eval()


def check_resume(args, students):
    if not args.resume:
        return None
    path = args.resume
    complete = json.loads((path/'complete.json').read_text())
    train.check_model_config(json.loads((path/'config.json').read_text()), args)
    if complete['data_manifest_sha256'] != file_hash(args.data/'manifest.json'):
        raise ValueError('Resume data differs')
    for name, expected in [('adapter.pt', complete['adapter_sha256']),
                           ('training_state.pt', complete['training_state_sha256']),
                           *complete.get('optimizer_shards', {}).items()]:
        if Path(name).name != name or file_hash(path/name) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    state = torch.load(path/'training_state.pt', map_location='cpu', weights_only=True)
    if state.get('backend') != BACKEND:
        raise ValueError('Resume requires a checkpoint from the dedicated-teacher FSDP trainer')
    if state['world_size'] != students:
        raise ValueError('Resume requires the original student GPU count')
    expected_shards = {f'optimizer-rank-{rank:05d}.pt' for rank in range(students)}
    if set(complete.get('optimizer_shards', {})) != expected_shards:
        raise ValueError('Missing optimizer shard metadata')
    train.check_resume_signature(state['training_signature'], args)
    return state


def save_checkpoint(model, optimizer, args, step, cursor, elapsed, students, rank, group):
    root = SETTINGS.paths['CHECKPOINT_DIR']/args.run_name
    path = root/f'step-{step:07d}'
    plan = [None]
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if path.exists() and checkpoint_metadata(path) is None:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f'Unexpected checkpoint path: {path}')
            path.rename(root/f'{path.name}.incomplete-{uuid4().hex[:12]}')
        if path.exists():
            meta = checkpoint_metadata(path)
            if meta['step'] != step or Path(meta['source_run']).resolve() != args.run_dir.resolve():
                raise RuntimeError(f'Checkpoint belongs to a different run: {path}')
            plan[0] = None
        else:
            temporary = root/f'.{path.name}-{uuid4().hex[:12]}.tmp'
            temporary.mkdir()
            plan[0] = str(temporary)
    dist.broadcast_object_list(plan, src=0, group=group)
    if plan[0] is not None:
        temporary = Path(plan[0])
        rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        rngs = [None]*students if rank == 0 else None
        dist.gather_object(rng, rngs, dst=0, group=group)
        torch.save(local_optimizer_state(optimizer), temporary/f'optimizer-rank-{rank:05d}.pt')
        weights = full_adapter_state(model.student, rank)
        if rank == 0:
            torch.save(weights, temporary/'adapter.pt')
        del weights
        dist.barrier(group=group)
        if rank == 0:
            torch.save({'backend': BACKEND, 'step': step, 'epoch': cursor.epoch,
                        'batch_in_epoch': cursor.batch_in_epoch, 'elapsed_seconds': elapsed,
                        'rngs': rngs, 'world_size': students,
                        'training_signature': train.training_signature(args),
                        'run_start_unix': args.run_start_unix}, temporary/'training_state.pt')
            write_json(temporary/'config.json', train.model_config(args))
            write_json(temporary/'complete.json', {
                'step': step, 'source_run': str(args.run_dir), 'elapsed_seconds': elapsed,
                'adapter_sha256': file_hash(temporary/'adapter.pt'),
                'training_state_sha256': file_hash(temporary/'training_state.pt'),
                'data_manifest_sha256': file_hash(args.data/'manifest.json'),
                'optimizer_shards': {p.name: file_hash(p) for p in sorted(temporary.glob('optimizer-rank-*.pt'))}})
            temporary.rename(path)
    if rank == 0:
        train.emit(args.run_dir, 'checkpoints.jsonl', {'step': step, 'path': str(path), 'elapsed_seconds': elapsed})
        train.prune_checkpoints(root, args.run_dir, args.max_checkpoints)
        print(f'checkpoint step={step} path={path}', flush=True)
    dist.barrier(group=group)
    return path


def validation_batches(data, order, rank, students, batch_size, sequence_length, pad_token_id):
    """Equal number and shape of forwards on every rank, without scoring duplicates."""
    for start in range(0, len(order), students*batch_size):
        indices = order[start+rank*batch_size:start+(rank+1)*batch_size]
        rows = [data[i] for i in indices]
        ids = torch.full((batch_size, sequence_length+1), pad_token_id, dtype=torch.long)
        lengths = torch.ones(batch_size, dtype=torch.long)  # Dummy rows: zero scored targets.
        if rows:
            real_ids, real_lengths = collate_tokens(rows, sequence_length=sequence_length, pad_token_id=pad_token_id)
            ids[:len(rows)], lengths[:len(rows)] = real_ids, real_lengths
        yield ids, lengths


@torch.no_grad()
def validate(model, channel, batches, device, group):
    totals = torch.zeros(model.stats_size, device=device, dtype=torch.float64)
    for ids, lengths in batches:
        ids, lengths = ids.to(device), lengths.to(device)
        teacher_hidden = channel.hidden(ids[:, :-1])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, stats = model(ids, lengths, teacher_hidden)
        if not torch.isfinite(stats).all().item():
            raise FloatingPointError('Non-finite validation metrics')
        totals += stats.double()
    dist.all_reduce(totals, group=group)
    if totals[2].item() <= 0:
        raise ValueError('Validation has no scored tokens')
    return {**loss_metrics(totals, model.ce_weight, model.hidden_mse_weight),
            'tokens': int(totals[2].item())}


@torch.no_grad()
def record_samples(model, tokenizer, args, step, rank, group, device, phase):
    """All students take identical forwards; rank-zero tokens determine EOS/order."""
    results = []
    prompts = train.sample_prompts()
    log_progress(args, rank, 'samples_start', step=step, phase=phase,
                 total_samples=len(prompts), max_new_tokens=args.sample_max_new_tokens)
    for index, item in enumerate(prompts, 1):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': item['prompt']}],
            tokenize=True, add_generation_prompt=True, reasoning_effort='low',
            conversation_start_date='2026-11-02', return_tensors='pt', return_dict=True).input_ids.to(device)
        tick, state, generated = time.perf_counter(), None, []
        context = {'step': step, 'phase': phase, 'sample': index,
                   'total_samples': len(prompts), 'id': item['id']}
        log_progress(args, rank, 'sample_start', **context, input_tokens=ids.shape[1])
        for start in range(0, ids.shape[1], 128):
            logits, state = model.student(ids[:, start:start+128], state, logits_to_keep=1)
        log_progress(args, rank, 'sample_prefill_complete', **context,
                     seconds=time.perf_counter()-tick)
        reason = 'length'
        eos = model.student.config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for _ in range(args.sample_max_new_tokens):
            token = logits[:, -1].argmax(-1)
            dist.broadcast(token, src=0, group=group)
            generated.append(token.item())
            if len(generated) == 1 or len(generated) % 16 == 0:
                log_progress(args, rank, 'sample_progress', **context,
                             output_tokens=len(generated), seconds=time.perf_counter()-tick)
            if token.item() in eos:
                reason = 'eos'
                break
            logits, state = model.student(token[:, None], state, logits_to_keep=1)
        if rank == 0:
            results.append({'id': item['id'], 'prompt': item['prompt'], 'token_ids': generated,
                'text': tokenizer.decode(generated, skip_special_tokens=False), 'stop_reason': reason,
                'input_tokens': ids.shape[1], 'output_tokens': len(generated), 'seconds': time.perf_counter()-tick})
            path = args.session_dir/'samples'/f'{phase}-step-{step:07d}.json'
            temporary = path.with_suffix('.json.tmp')
            write_json(temporary, {'step': step, 'phase': phase, 'samples': results,
                                  'complete': index == len(prompts), 'total_samples': len(prompts)})
            temporary.replace(path)
        log_progress(args, rank, 'sample_complete', **context,
                     output_tokens=len(generated), stop_reason=reason, seconds=time.perf_counter()-tick)
    dist.barrier(group=group)
    log_progress(args, rank, 'samples_complete', step=step, phase=phase)


def main():
    args = parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world < 2:
        raise ValueError('Need at least one student GPU and one teacher GPU')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    students = world-1
    group = dist.new_group(list(range(students)), timeout=timedelta(minutes=30))
    try:
        with ExitStack() as stack:
            info = [None]
            if rank == 0:
                try:
                    stack.enter_context(exclusive_run(args.run_dir, os.environ.get('LLM_JP_LOGKV_RUN_TOKEN')))
                    args.resume = resolve_resume(args, SETTINGS)
                    saved = check_resume(args, students)
                    session = os.environ.get('LLM_JP_LOGKV_SESSION_ID') or new_session_id()
                    create_session(args.run_dir, session)
                    info[0] = {'resume': str(args.resume) if args.resume else None, 'session': session}
                except (OSError, ValueError, RuntimeError) as exc:
                    info[0] = {'error': str(exc)}
            dist.broadcast_object_list(info, src=0)
            if 'error' in info[0]:
                raise RuntimeError(info[0]['error'])
            args.resume = Path(info[0]['resume']) if info[0]['resume'] else None
            args.session_dir = args.run_dir/'sessions'/info[0]['session']
            # Read state metadata on students only. Rank zero already verified all hashes.
            saved = (torch.load(args.resume/'training_state.pt', map_location='cpu', weights_only=True)
                     if args.resume and rank < students else None)
            run_training(args, device, rank, students, group, saved)
    finally:
        dist.destroy_process_group()


def run_training(args, device, rank, students, group, saved):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    stop_requested = {'value': False}
    if rank < students:
        def stop(signum, frame):
            stop_requested['value'] = True
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    if rank == 0:
        (args.session_dir/'samples').mkdir()
        data_manifest = json.loads((args.data/'manifest.json').read_text())
        if data_manifest['tokenizer_revision'] != SETTINGS.model_revision:
            raise ValueError('Token cache uses a different model revision')
        for name, expected in data_manifest['files'].items():
            if file_hash(args.data/name) != expected:
                raise ValueError(f'Token cache checksum mismatch: {name}')
        write_json(args.session_dir/'manifest.json', {
            'backend': BACKEND, 'created_at': datetime.now(timezone.utc).isoformat(),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
            'source_files_sha256': {str(p): file_hash(p) for p in sorted(Path('src').rglob('*.py'))},
            'uv_lock_sha256': file_hash(Path('uv.lock')), 'model': train.model_config(args),
            'data_manifest': data_manifest, 'data_manifest_sha256': file_hash(args.data/'manifest.json'),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'student_world_size': students, 'teacher_global_rank': students,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
            'teacher_backend': 'sdpa; sequential per-student batches; final normalized BF16 hidden transfer; all raw block states when hidden_mse_weight > 0',
            'sharding': 'FSDP2 trainable attention and conv only; replicated frozen student modules',
            'optimizer': 'AdamW FP32 master/moments, foreach=False, betas=(0.9,0.95), weight_decay=0.01',
            'objective': 'Exact full-vocabulary KL(teacher || student)*T^2 + ce_weight*next-token CE + hidden_mse_weight*mean all-block MSE',
            'gradient_accumulation': 'Reduce-scatter every microbatch; global nonpadding-token normalization'})
    dist.barrier()
    teacher = load_source(device) if rank == students else None
    # Avoid simultaneously materializing five full CPU initialization models.
    for initializing in range(students):
        if rank == initializing:
            source = load_source('cpu')
            torch.manual_seed(args.seed)
            model = RemoteStudent(source, chunk_size=args.chunk_size, no_rope=args.no_rope,
                conv_kernel_size=args.conv_kernel_size, loss_chunk_tokens=args.loss_chunk_tokens,
                temperature=args.temperature, ce_weight=args.ce_weight,
                hidden_mse_weight=args.hidden_mse_weight)
            del source
            if args.resume:
                load_adapter(model.student, torch.load(args.resume/'adapter.pt', map_location='cpu', weights_only=True))
            model.to(device)
            gc.collect()
        dist.barrier()
    if rank == students:
        channel = TeacherChannel(device, students, teacher.config.hidden_size,
                                 teacher.config.num_hidden_layers if args.hidden_mse_weight else 0)
        torch.cuda.reset_peak_memory_stats(device)
        channel.serve(teacher)
        write_json(args.session_dir/'teacher-completed.json', {
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'global_rank': rank})
        dist.barrier()
        return

    model = shard_student(model, group)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(.9, .95), weight_decay=.01, foreach=False)
    step = epoch = batch_in_epoch = 0
    if saved:
        restore_local_optimizer(optimizer, torch.load(args.resume/f'optimizer-rank-{rank:05d}.pt',
                                                      map_location='cpu', weights_only=True))
        step, epoch, batch_in_epoch = saved['step'], saved['epoch'], saved['batch_in_epoch']
        args.run_start_unix = saved['run_start_unix']
        torch.set_rng_state(saved['rngs'][rank]['cpu'])
        torch.cuda.set_rng_state(saved['rngs'][rank]['cuda'])
    else:
        start = [time.time() if rank == 0 else None]
        dist.broadcast_object_list(start, src=0, group=group)
        args.run_start_unix = start[0]
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model_path, local_files_only=True)
    channel = TeacherChannel(device, students, model.student.config.hidden_size,
                             len(model.student.layers) if args.hidden_mse_weight else 0)
    data = TokenCache(args.data, 'train', args.sequence_length)
    valid = TokenCache(args.data, 'valid', args.sequence_length)
    sampler = DistributedSampler(data, num_replicas=students, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True)
    from functools import partial
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, drop_last=True,
        collate_fn=partial(collate_tokens, sequence_length=args.sequence_length, pad_token_id=tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(args.seed), pin_memory=True, num_workers=0)
    cursor = train.EpochBatchIterator(loader, sampler, epoch, batch_in_epoch)
    val_order = np.random.default_rng(args.seed+1).permutation(len(valid)).tolist()

    def validation(phase, all_examples=False):
        order = val_order if all_examples else val_order[:students*args.val_examples_per_rank]
        tick = time.perf_counter()
        log_progress(args, rank, 'validation_start', step=step, phase=phase, examples=len(order))
        batches = validation_batches(valid, order, rank, students, args.batch_size,
                                     args.sequence_length, tokenizer.pad_token_id)
        values = validate(model, channel, batches, device, group)
        if rank == 0:
            train.emit(args.run_dir, 'validation.jsonl', {'step': step, 'phase': phase, **values})
        log_progress(args, rank, 'validation_complete', step=step, phase=phase,
                     seconds=time.perf_counter()-tick, **values)
        return values

    def elapsed():
        return time.time()-args.run_start_unix

    if rank == 0:
        write_json(args.session_dir/'runtime.json', {
            'student_gpus': students, 'teacher_gpus': 1,
            'trainable_parameters': sum(p.numel() for p in params),
            'train_examples': len(data), 'valid_examples': len(valid),
            'global_examples_per_step': students*args.batch_size*args.grad_accum,
            'batches_per_epoch_per_rank': len(loader),
            'optimizer_steps_per_epoch': math.ceil(len(loader)/args.grad_accum),
            'num_epochs': args.num_epochs, 'max_hours': args.max_hours,
            'frozen_student_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'resident_trainable_shard_parameters': sum(p.to_local().numel() for p in params)})
        write_json(args.session_dir/'started.json', {'step': step, 'epoch': cursor.epoch,
            'batch_in_epoch': cursor.batch_in_epoch, 'started_at': datetime.now(timezone.utc).isoformat()})
        print(f'FSDP students={students}, dedicated teacher=1; epochs={args.num_epochs}, max_hours={args.max_hours}', flush=True)
    # Honor a pending stop before expensive initial validation or generation.
    _, reason, _ = train.training_control(args, step, elapsed(), stop_requested['value'], device,
        epoch=cursor.epoch, batch_in_epoch=cursor.batch_in_epoch, batches_per_epoch=len(loader), group=group)
    if reason is None:
        values = validation('initial')
        if not args.skip_samples and not args.skip_initial_samples:
            record_samples(model, tokenizer, args, step, rank, group, device, 'initial')
    log_progress(args, rank, 'training_loop_start', step=step, stop_reason=reason)
    while reason is None:
        _, reason, lr = train.training_control(args, step, elapsed(), stop_requested['value'], device,
            epoch=cursor.epoch, batch_in_epoch=cursor.batch_in_epoch, batches_per_epoch=len(loader), group=group)
        if reason is not None:
            break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        batches = cursor.next_update(args.grad_accum)
        count = torch.tensor(sum(int((lengths-1).sum()) for _, lengths in batches), device=device, dtype=torch.float64)
        dist.all_reduce(count, group=group)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        totals = torch.zeros(model.stats_size, device=device, dtype=torch.float64)
        for ids, lengths in batches:
            ids, lengths = ids.to(device), lengths.to(device)
            teacher_hidden = channel.hidden(ids[:, :-1])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, stats = model(ids, lengths, teacher_hidden)
                scaled = loss*(students/count)
            finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
            if not finite.item():
                raise FloatingPointError('Non-finite distributed loss')
            scaled.backward()  # Synchronize every microbatch, retaining only sharded gradients.
            totals += stats.double()
            del teacher_hidden, loss, scaled, stats
        norm = clip_sharded_grad_norm(params, group)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(totals, group=group)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter()-tick
        step += 1
        peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            metrics = {'step': step, 'elapsed_seconds': elapsed(), 'lr': lr,
                **loss_metrics(totals, args.ce_weight, args.hidden_mse_weight),
                'grad_norm': norm.item(), 'valid_tokens': int(totals[2].item()),
                'step_seconds': seconds, 'valid_tokens_per_second': totals[2].item()/seconds,
                'epoch': cursor.epoch, 'batch_in_epoch': cursor.batch_in_epoch,
                'microbatches_per_rank': len(batches), 'max_rank_peak_allocated_bytes': int(peak.item())}
            train.emit(args.run_dir, 'metrics.jsonl', metrics)
            print(json.dumps(metrics), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(model, optimizer, args, step, cursor, elapsed(), students, rank, group)
        if step % args.validate_every == 0:
            values = validation('periodic')
        if not args.skip_samples and step % args.sample_every == 0:
            record_samples(model, tokenizer, args, step, rank, group, device, 'periodic')
    final_path = save_checkpoint(model, optimizer, args, step, cursor, elapsed(), students, rank, group)
    values = None
    if reason != 'save_and_exit':
        if not args.skip_final_validation:
            values = validation('final_all_heldout', all_examples=True)
        if not args.skip_samples:
            record_samples(model, tokenizer, args, step, rank, group, device, 'final')
    channel.command(channel.STOP)
    dist.barrier()
    if rank == 0:
        write_json(args.session_dir/'completed.json', {'backend': BACKEND, 'step': step, 'reason': reason,
            'epoch': cursor.epoch, 'batch_in_epoch': cursor.batch_in_epoch, 'elapsed_seconds': elapsed(),
            'checkpoint': str(final_path), 'validation': values,
            'completed_at': datetime.now(timezone.utc).isoformat()})
        print(f'Training finished: {reason}; checkpoint={final_path}', flush=True)


if __name__ == '__main__':
    main()
