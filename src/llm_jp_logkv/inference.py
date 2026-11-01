"""Local pretrained inference and small, recorded no-distillation comparisons."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import math
import os
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

# This module is imported by CLI only after load_settings.
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from .llama_model import LlamaLogKVForCausalLM


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def new_run(settings, kind):
    path = settings.paths['EXPERIMENT_DIR'] / f'{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def provenance(settings, args):
    root = Path(__file__).resolve().parents[2]
    def git(*words):
        return subprocess.check_output(['git', '-C', str(root), *words], text=True).strip()
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_commit': git('rev-parse', 'HEAD'),
        'source_dirty': bool(git('status', '--porcelain')),
        'model_id': settings.model_id, 'model_revision': settings.model_revision,
        'model_path': str(settings.model_path),
        'packages': {name: version(name) for name in ('torch', 'transformers', 'huggingface-hub', 'accelerate')},
        'lock_sha256': file_hash(root / 'uv.lock'),
        'model_files': {p.name: {'bytes': p.stat().st_size, **(
            {'sha256': file_hash(p)} if p.suffix != '.safetensors' else {})}
            for p in sorted(settings.model_path.iterdir()) if p.is_file()},
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'training': False, 'distillation': False,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device': args.device, 'dtype': 'bfloat16',
        'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
        'attention': {'all_layers': True, 'chunk_size': args.chunk_size,
                      'self_slot': True, 'rope': 'original, applied once before pooling',
                      'compressed_cache_heads': 32, 'extra_conv_gate_norm': False},
    }


def load_model(settings, device):
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path, local_files_only=True)
    model, info = LlamaForCausalLM.from_pretrained(
        settings.model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': device}, attn_implementation='eager', output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Pretrained weight loading was not exact: {info}')
    return tokenizer, model.eval()


def prompt_ids(tokenizer, prompt, args, *, date):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True,
        reasoning_effort=args.reasoning_effort, conversation_start_date=date,
        return_tensors='pt', return_dict=True).input_ids.to(args.device)


def sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, tokenizer, ids, args, mode):
    sync(args.device)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    if mode == 'original':
        output = model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=args.max_new_tokens, eos_token_id=model.config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True)
        tokens = output.sequences[:, ids.shape[1]:]
        eos = model.config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        reason = 'eos' if tokens[0, -1].item() in eos else 'length'
        state_bytes = None
    else:
        tokens, state, reason = model.greedy_generate(
            ids, max_new_tokens=args.max_new_tokens,
            eos_token_id=model.config.eos_token_id, prefill_chunk_size=args.prefill_chunk_size)
        state_bytes = state.tensor_bytes
    sync(args.device)
    seconds = time.perf_counter() - started
    return {
        'input_tokens': ids.shape[1], 'output_tokens': tokens.shape[1],
        'token_ids': tokens[0].tolist(), 'text': tokenizer.decode(tokens[0], skip_special_tokens=False),
        'text_without_special_tokens': tokenizer.decode(tokens[0], skip_special_tokens=True),
        'stop_reason': reason, 'seconds_including_prefill': seconds,
        'logical_state_tensor_bytes': state_bytes,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device)
            if args.device.startswith('cuda') else None,
    }


@torch.inference_mode()
def text_nll(model, ids, mode, chunk_size):
    """Token-weighted next-token NLL of plain text, BOS included, no chat.

    Full-vocabulary logits are converted to FP32 only for a bounded segment.
    All tokens after BOS are scored; no overlap or boundary token is omitted.
    """
    total, count, state = 0., 0, None
    for start in range(0, ids.shape[1] - 1, chunk_size):
        end = min(start + chunk_size, ids.shape[1] - 1)
        if mode == 'original':
            out = model(ids[:, start:end], past_key_values=state, use_cache=True)
            logits, state = out.logits, out.past_key_values
        else:
            logits, state = model(ids[:, start:end], state)
        target = ids[:, start + 1:end + 1]
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction='sum')
        total += loss.item()
        count += target.numel()
    return {'nll_sum': total, 'tokens': count, 'mean_nll': total / count}


@torch.inference_mode()
def dense_check(source, ids):
    wrapper = LlamaLogKVForCausalLM(source, mode='dense')
    expected = source(ids, use_cache=False).logits.float()
    actual, _ = wrapper(ids)
    delta = (actual.float() - expected).abs()
    result = {'tokens': ids.shape[1], 'max_abs_logit_error': delta.max().item(),
              'mean_abs_logit_error': delta.mean().item(),
              'argmax_agreement': (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()}
    # Same eager backend and BF16 projections: full-sequence execution should
    # agree. Explicit loose low-precision tolerance, independent of quality.
    torch.testing.assert_close(actual.float(), expected, atol=0.125, rtol=0.02)
    result['passed'] = True
    return result


def run_infer(settings, args):
    run = new_run(settings, 'inference')
    print(f'Artifact directory: {run}', flush=True)
    manifest = provenance(settings, args)
    write_json(run / 'manifest.json', manifest)
    tokenizer, source = load_model(settings, args.device)
    model = source if args.mode == 'original' else LlamaLogKVForCausalLM(
        source, mode='logkv', chunk_size=args.chunk_size, no_level_bias=args.no_level_bias)
    ids = prompt_ids(tokenizer, args.prompt, args, date=args.conversation_start_date)
    result = generate(model, tokenizer, ids, args, args.mode)
    write_json(run / 'result.json', {'prompt': args.prompt, 'mode': args.mode, **result})
    return {'artifact_dir': str(run), **result}


def run_evaluation(settings, args):
    run = new_run(settings, 'initial-quality')
    print(f'Artifact directory: {run}', flush=True)
    suite = json.loads(args.suite.read_text(encoding='utf-8'))
    manifest = provenance(settings, args)
    manifest['suite_sha256'] = file_hash(args.suite)
    manifest['suite'] = suite
    manifest['limitations'] = 'Handwritten smoke set; not a benchmark or long-context evaluation. No training.'
    write_json(run / 'manifest.json', manifest)
    tokenizer, source = load_model(settings, args.device)
    ids = prompt_ids(tokenizer, suite['prompts'][0]['prompt'], args, date=args.conversation_start_date)
    write_json(run / 'dense-equivalence.json', dense_check(source, ids))
    results = []
    with (run / 'results.jsonl').open('x', encoding='utf-8') as stream:
        for mode in ('original', 'logkv', 'logkv_no_bias'):
            model = source if mode == 'original' else LlamaLogKVForCausalLM(
                source, chunk_size=args.chunk_size, no_level_bias=mode == 'logkv_no_bias')
            for item in suite['prompts']:
                ids = prompt_ids(tokenizer, item['prompt'], args, date=args.conversation_start_date)
                result = {'kind': 'generation', 'mode': mode, **item,
                          **generate(model, tokenizer, ids, args, mode)}
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                print(f"{mode}/{item['id']}: {result['input_tokens']}+{result['output_tokens']} tokens, "
                      f"{result['stop_reason']}, {result['seconds_including_prefill']:.1f}s", flush=True)
                results.append(result)
            for item in suite['texts']:
                # Preserve original BOS behavior and never downcast token IDs.
                ids = tokenizer(item['text'], add_special_tokens=True, return_tensors='pt').input_ids.to(args.device)
                result = {'kind': 'nll', 'mode': mode, **item,
                          **text_nll(model, ids, mode, args.prefill_chunk_size)}
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                results.append(result)
                print(f"{mode}/{item['id']}: NLL={result['mean_nll']:.4f}, {result['tokens']} tokens", flush=True)
            if mode != 'original':
                del model
    summary = {}
    for mode in ('original', 'logkv', 'logkv_no_bias'):
        nlls = [r for r in results if r['mode'] == mode and r['kind'] == 'nll']
        mean = sum(r['nll_sum'] for r in nlls) / sum(r['tokens'] for r in nlls)
        summary[mode] = {'mean_nll': mean, 'perplexity': math.exp(mean),
                         'tokens': sum(r['tokens'] for r in nlls),
                         'generation_eos_count': sum(r['stop_reason'] == 'eos' for r in results
                                                     if r['mode'] == mode and r['kind'] == 'generation')}
    write_json(run / 'summary.json', summary)
    return {'artifact_dir': str(run), 'summary': summary}
