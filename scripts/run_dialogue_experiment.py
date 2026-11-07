#!/usr/bin/env python3
"""Run a persistent LogKV dialogue from a queue of numbered prompt JSON files.

The partner writes prompts to RUN/prompts/000001.json, etc. This process never
invents partner replies. It records full analysis/final text, sampled token IDs,
timings, CUDA memory and compressed state snapshots under a new RAID run directory.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import signal
import subprocess
import threading
import time
from unittest.mock import patch

from llm_jp_logkv.settings import load_settings


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def cpu_tree(value):
    """Copy cache tensors without changing nested containers or integer offsets."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--target-output-tokens', type=int, default=100000)
    parser.add_argument('--max-new-tokens', type=int, default=3072)
    parser.add_argument('--max-hours', type=float, default=7.5)
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-p', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    if args.target_output_tokens < 1 or args.max_new_tokens < 1 or not 0 < args.max_hours < 8:
        parser.error('Positive token limits and a time budget below 8 hours are required')
    run = args.run_dir.resolve()
    if not run.is_relative_to(settings.paths['EXPERIMENT_DIR'].resolve()):
        parser.error('--run-dir must be within EXPERIMENT_DIR')
    run.mkdir(parents=True, exist_ok=True)
    if (run/'manifest.json').exists():
        parser.error('Experiment already exists; choose a new directory')
    for name in ('prompts', 'turns', 'states'):
        (run/name).mkdir(exist_ok=True)

    import torch
    from llm_jp_logkv import chat_server
    from llm_jp_logkv.inference import load_model, file_hash
    from llm_jp_logkv.distillation import student_from_checkpoint

    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint/'config.json').read_text())
    root = Path(__file__).resolve().parents[1]
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'checkpoint': str(checkpoint), 'checkpoint_config': config,
        'adapter_sha256': file_hash(checkpoint/'adapter.pt'),
        'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()),
        'script_sha256': file_hash(Path(__file__)), 'lock_sha256': file_hash(root/'uv.lock'),
        'chat_engine_sha256': file_hash(root/'src/llm_jp_logkv/chat_server.py'),
        'target_definition': 'Model sampled output tokens, including analysis and EOS; excludes input and synthetic boundaries',
        'partner': 'Codex-authored prompt files; no external LLM API calls',
        'gpu': torch.cuda.get_device_name(0), 'training': False,
    }
    write_json(run/'manifest.json', manifest)
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancel.set())
    started = time.monotonic()
    tokenizer, source = load_model(settings, 'cuda:0')
    model = student_from_checkpoint(source, checkpoint, model_id=settings.model_id,
                                    model_revision=settings.model_revision, chunk_size=config['chunk_size'])
    del source
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine_args = chat_server.parse_args(['--device', 'cuda:0', '--max-new-tokens', str(args.max_new_tokens)])
    engine = chat_server.ChatEngine(model, tokenizer, engine_args)
    session = chat_server.ChatSession()
    total_input = total_output = index = 0
    saved_at = -1
    status = 'waiting_for_prompt'
    sampler = chat_server.sample_token
    deadline = started + args.max_hours * 3600

    def progress(**extra):
        write_json(run/'progress.json', {
            'status': status, 'turn': index, 'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'session_tokens': session.state.offset if session.state else 0,
            'elapsed_seconds': time.monotonic() - started, **extra})

    def save_state():
        nonlocal saved_at
        if session.state is None or saved_at == session.state.offset:
            return
        offset = session.state.offset
        payload = {'offset': offset, 'turn': index, 'effort': session.effort,
                   'total_input_tokens': total_input, 'total_output_tokens': total_output,
                   'adapter_sha256': manifest['adapter_sha256'],
                   'conv_caches': cpu_tree(session.state.conv_caches),
                   'layers': [{'mode': s.mode, 'offset': s.offset, 'payload': cpu_tree(s.payload)}
                              for s in session.state.layers],
                   'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        destination = run/'states'/f'turn-{index:06d}-offset-{offset:09d}.pt'
        temporary = destination.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(destination)
        saved_at = offset

    progress()
    try:
        while total_output < args.target_output_tokens and not cancel.is_set():
            if time.monotonic() >= deadline:
                status = 'time_budget'
                break
            control = run/'control.cmd'
            if control.exists() and control.read_text().strip() in ('stop', 'save_and_exit'):
                status = 'operator_stop'
                break
            prompt_path = run/'prompts'/f'{index+1:06d}.json'
            if not prompt_path.exists():
                progress()
                cancel.wait(1)
                continue
            spec = json.loads(prompt_path.read_text())
            prompt = spec['prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'Empty prompt: {prompt_path}')
            options = {'reasoning_effort': args.reasoning_effort, 'temperature': args.temperature,
                       'top_p': args.top_p, 'max_new_tokens': min(spec.get('max_new_tokens', args.max_new_tokens),
                           args.max_new_tokens, args.target_output_tokens-total_output)}
            index += 1
            turn_dir = run/'turns'/f'{index:06d}'
            turn_dir.mkdir(exist_ok=False)
            write_json(turn_dir/'request.json', {'prompt': prompt, 'options': options, 'metadata': spec})
            prepared = engine.prepare(prompt, options, session)
            write_json(turn_dir/'input-token-ids.json', prepared.ids[0].tolist())
            status = 'generating'
            progress()
            sampled = []
            last_progress = 0.
            torch.cuda.reset_peak_memory_stats()
            def sample(*values):
                token = sampler(*values)
                sampled.append(token)
                return token
            with patch.object(chat_server, 'sample_token', sample):
                for event in engine.stream(prepared, options, cancel):
                    now = time.monotonic()
                    if now >= deadline or (control.exists() and control.read_text().strip() in ('stop', 'save_and_exit')):
                        cancel.set()
                    if now-last_progress >= 3 or event['type'] == 'done':
                        last_progress = now
                        write_json(run/'current-turn.json', {'turn': index, 'prompt': prompt, **event})
                        progress(current_output_tokens=len(sampled),
                                 cuda_allocated_bytes=torch.cuda.memory_allocated())
                        write_json(turn_dir/'sampled-token-ids.json', sampled)
                    if event['type'] == 'done':
                        result = event
            if len(sampled) != result['output_tokens']:
                raise RuntimeError('Sampled token accounting mismatch')
            if not prepared.retained:
                status = 'input_cancelled'
                break
            session.state, session.effort = prepared.result_state, args.reasoning_effort
            total_input += result['input_tokens']
            total_output += result['output_tokens']
            result.update(total_input_tokens=total_input, total_output_tokens=total_output,
                          state_tensor_bytes=session.state.tensor_bytes,
                          cuda_allocated_bytes=torch.cuda.memory_allocated(),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            write_json(turn_dir/'result.json', result)
            with (run/'transcript.md').open('a') as stream:
                stream.write(f'\n## Turn {index}: Codex\n\n{prompt}\n\n')
                for part in result['parts']:
                    stream.write(f"### LogKV {part['channel']}\n\n{part['text']}\n\n")
                stream.write(f"State: {result['session_tokens']}; sampled total: {total_output}; stop: {result['stop_reason']}\n")
            with (run/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({k: v for k, v in result.items() if k not in ('parts', 'raw', 'final')})+'\n')
            if saved_at < 0 or session.state.offset-saved_at >= 10000:
                save_state()
            status = 'waiting_for_prompt'
            progress()
            print(f'Turn {index}: {result["input_tokens"]}+{result["output_tokens"]}; '
                  f'output_total={total_output}; state={session.state.offset}; {result["stop_reason"]}', flush=True)
        if total_output >= args.target_output_tokens:
            status = 'target_reached'
        elif cancel.is_set() and status != 'input_cancelled':
            status = 'time_budget' if time.monotonic() >= deadline else 'operator_stop'
    except Exception as exc:
        status = 'error'
        write_json(run/'error.json', {'error': repr(exc), 'turn': index})
        raise
    finally:
        save_state()
        progress()
        write_json(run/'completed.json', json.loads((run/'progress.json').read_text()))


if __name__ == '__main__':
    main()
