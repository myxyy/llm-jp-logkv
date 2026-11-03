"""Local, single-model chat server. No HF imports until settings are loaded.

POST /api/chat streams newline-delimited JSON. The browser owns its conversation;
each turn uses the original chat template and a fresh LogKV state. Inference is
serialized, with cancellation checked between prefill chunks and decode steps.
"""
import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .training_runs import checkpoint_metadata, completed_checkpoints, named_directory


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', help='Select a run under CHECKPOINT_DIR')
    parser.add_argument('--checkpoint', default='latest',
                        help='latest (requires --run-name), step name, or checkpoint directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda:0', help='cuda:0, cuda:1, cpu, or GPU index')
    parser.add_argument('--max-new-tokens', type=positive_int, default=2048,
                        help='Default and maximum output length per turn, including analysis')
    parser.add_argument('--max-input-tokens', type=positive_int, default=8192,
                        help='Reject longer histories; never silently truncate')
    parser.add_argument('--prefill-chunk-size', type=positive_int, default=128)
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], default='low')
    parser.add_argument('--temperature', type=float, default=0., help='0 = greedy; range 0–2')
    parser.add_argument('--top-p', type=float, default=1., help='Range (0, 1]')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error('temperature must be in [0, 2] and top-p in (0, 1]')
    if args.device.isdecimal():
        args.device = 'cuda:' + args.device
    return args


def resolve_checkpoint(settings, args):
    root = named_directory(settings, 'CHECKPOINT_DIR', args.run_name) if args.run_name else None
    if args.checkpoint == 'latest':
        if root is None:
            raise ValueError('--checkpoint latest requires --run-name')
        candidates = completed_checkpoints(root)
        if not candidates:
            raise FileNotFoundError(f'No completed checkpoints in {root}')
        path = candidates[-1]
    else:
        path = Path(args.checkpoint).expanduser()
        if root is not None and path.parent == Path('.'):
            path = root / path
    path = path.resolve()
    if checkpoint_metadata(path) is None:
        raise ValueError(f'Not a completed checkpoint: {path}')
    return path


def validate_request(body, defaults):
    if not isinstance(body, dict):
        raise ValueError('Expected a JSON object')
    request_id = body.get('request_id')
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError('request_id must be a nonempty string of at most 128 characters')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages or len(messages) > 201:
        raise ValueError('messages must contain 1–201 user/assistant messages')
    clean = []
    for i, message in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        if (not isinstance(message, dict) or message.get('role') != role
                or not isinstance(message.get('content'), str) or not message['content'].strip()):
            raise ValueError('Alternate nonempty user/assistant messages, starting and ending with user')
        clean.append({'role': role, 'content': message['content']})
    if len(clean) % 2 == 0:
        raise ValueError('The last message must be user')
    options = {}
    for key, lower, upper in [('temperature', 0, 2), ('top_p', 0, 1)]:
        value = body.get(key, getattr(defaults, key))
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not lower <= value <= upper or (key == 'top_p' and value == 0)):
            raise ValueError(f'Invalid {key}')
        options[key] = value
    limit = body.get('max_new_tokens', defaults.max_new_tokens)
    if type(limit) is not int or not 1 <= limit <= defaults.max_new_tokens:
        raise ValueError(f'max_new_tokens must be 1–{defaults.max_new_tokens}')
    effort = body.get('reasoning_effort', defaults.reasoning_effort)
    if effort not in ('low', 'medium', 'high'):
        raise ValueError('Invalid reasoning_effort')
    return request_id, clean, dict(options, max_new_tokens=limit, reasoning_effort=effort)


class HarmonyOutput:
    """Parse structural token IDs, not possibly incomplete decoded delimiters.

    Decode whole accumulated bodies so Japanese byte-fallback pieces are never
    decoded individually. Snapshots may replace a previous incomplete character.
    Unknown/malformed headers remain available in the raw output.
    """
    def __init__(self, tokenizer, eos_token_ids=()):
        self.tokenizer = tokenizer
        self.special = {tokenizer.convert_tokens_to_ids(f'<|{name}|>'): name
                        for name in ('start', 'channel', 'message', 'end', 'return')}
        for token in eos_token_ids:
            self.special.setdefault(token, 'return')
        self.tokens, self.parts, self.header = [], [], []
        self.in_body = False
        self.has_channel = False

    def push(self, token):
        self.tokens.append(token)
        kind = self.special.get(token)
        if kind in ('start', 'end', 'return'):
            self.in_body, self.has_channel = False, False
            self.header = []
        elif kind == 'channel':
            self.in_body, self.has_channel = False, True
            self.header = []
        elif kind == 'message':
            header = self.decode(self.header).strip().split()
            channel = header[0] if self.has_channel and header else 'final'
            self.parts.append({'channel': channel, 'tokens': []})
            self.in_body = True
        elif self.in_body:
            self.parts[-1]['tokens'].append(token)
        else:
            self.header.append(token)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def snapshot(self, *, finished=False):
        def decoded(tokens):
            text = self.decode(tokens)
            return text if finished else text.rstrip('\ufffd')
        return {'parts': [{'channel': part['channel'], 'text': decoded(part['tokens'])}
                          for part in self.parts], 'raw': decoded(self.tokens)}


def sample_token(logits, temperature, top_p):
    import torch
    logits = logits.float()
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError('Non-finite generation logits')
    if temperature == 0:
        return logits.argmax().item()
    probabilities = (logits / temperature).softmax(-1)
    if top_p < 1:
        probabilities, indices = probabilities.sort(descending=True)
        # Keep the first token crossing the threshold, including for tiny top_p.
        probabilities[probabilities.cumsum(-1) - probabilities >= top_p] = 0
        return indices[torch.multinomial(probabilities, 1)].item()
    return torch.multinomial(probabilities, 1).item()


class ChatEngine:
    def __init__(self, model, tokenizer, args):
        self.model, self.tokenizer, self.args = model, tokenizer, args
        self.conversation_date = date.today().isoformat()

    def prepare(self, messages, options):
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt',
            return_dict=True, reasoning_effort=options['reasoning_effort'],
            conversation_start_date=self.conversation_date).input_ids
        if ids.shape[1] > self.args.max_input_tokens:
            raise ValueError(f'Input has {ids.shape[1]} tokens; limit is {self.args.max_input_tokens}. '
                             'Start a new conversation or increase --max-input-tokens.')
        if ids.shape[1] + options['max_new_tokens'] > self.model.config.max_position_embeddings:
            raise ValueError('Input plus output limit exceeds model context length')
        return ids.to(self.args.device)

    def stream(self, ids, options, cancel):
        import torch
        # inference_mode must be entered in the HTTP worker thread.
        with torch.inference_mode():
            state, reason, count = None, 'length', 0
            eos = self.model.config.eos_token_id
            eos = set([eos] if isinstance(eos, int) else (eos or []))
            output = HarmonyOutput(self.tokenizer, eos)
            started = time.perf_counter()
            decode_started = None
            yield {'type': 'start', 'input_tokens': ids.shape[1]}
            for part in ids.split(self.args.prefill_chunk_size, dim=1):
                if cancel.is_set():
                    break
                logits, state = self.model(part, state, logits_to_keep=1)
                yield {'type': 'prefill', 'processed_tokens': state.offset, 'input_tokens': ids.shape[1]}
            if not cancel.is_set():
                decode_started = time.perf_counter()
                for _ in range(options['max_new_tokens']):
                    if cancel.is_set():
                        break
                    token = sample_token(logits[0, -1], options['temperature'], options['top_p'])
                    count += 1
                    output.push(token)
                    yield {'type': 'update', 'output_tokens': count, **output.snapshot()}
                    if token in eos:
                        reason = 'eos'
                        break
                    if count < options['max_new_tokens'] and not cancel.is_set():
                        logits, state = self.model(ids.new_tensor([[token]]), state, logits_to_keep=1)
            if cancel.is_set():
                reason = 'cancelled'
            ended = time.perf_counter()
            snapshot = output.snapshot(finished=True)
            final = ''.join(p['text'] for p in snapshot['parts'] if p['channel'] == 'final')
            yield {'type': 'done', 'stop_reason': reason, 'input_tokens': ids.shape[1],
                   'output_tokens': count, 'seconds': ended - started,
                   'tokens_per_second': count / max(ended - decode_started, 1e-9) if decode_started else 0,
                   'final': final, 'complete': reason == 'eos' and bool(final.strip()), **snapshot}


class ChatApplication:
    def __init__(self, engine, args, checkpoint):
        self.engine, self.args = engine, args
        self.checkpoint = str(checkpoint)
        self.lock = threading.Lock()
        self.active = None

    def claim(self, request_id):
        with self.lock:
            if self.active is not None:
                return None
            cancel = threading.Event()
            self.active = (request_id, cancel)
            return cancel

    def stop(self, request_id=None):
        with self.lock:
            if self.active and (request_id is None or request_id == self.active[0]):
                self.active[1].set()
                return True
            return False

    def release(self):
        with self.lock:
            self.active = None


def make_server(app, host, port):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, value, content_type='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.respond(200, Path(__file__).with_name('chat.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/api/config':
                self.respond(200, {'checkpoint': app.checkpoint, **{
                    key: getattr(app.args, key) for key in
                    ('max_new_tokens', 'max_input_tokens', 'temperature', 'top_p', 'reasoning_effort')}})
            else:
                self.respond(404, {'error': 'Not found'})

        def read_body(self):
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                raise ValueError('Cross-origin requests are not supported')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('Content-Type must be application/json')
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1_000_000:
                raise ValueError('JSON body must be 1–1000000 bytes')
            return json.loads(self.rfile.read(size))

        def do_POST(self):
            if self.path not in ('/api/chat', '/api/stop'):
                self.respond(404, {'error': 'Not found'})
                return
            try:
                body = self.read_body()
                if self.path == '/api/stop':
                    if not isinstance(body, dict) or not isinstance(body.get('request_id'), str):
                        raise ValueError('request_id is required')
                    self.respond(200, {'stopped': app.stop(body['request_id'])})
                    return
                request_id, messages, options = validate_request(body, app.args)
            except (ValueError, OSError) as exc:
                self.respond(400, {'error': str(exc)})
                return
            cancel = app.claim(request_id)
            if cancel is None:
                self.respond(409, {'error': 'Another response is being generated. Retry after it finishes.'})
                return
            stream = None
            streaming = False
            try:
                ids = app.engine.prepare(messages, options)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                streaming = True
                stream = app.engine.stream(ids, options, cancel)
                for event in stream:
                    self.send_event(event)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                cancel.set()
            except Exception as exc:
                if streaming:
                    try:
                        self.send_event({'type': 'error', 'error': str(exc)})
                    except OSError:
                        pass
                else:
                    self.respond(400 if isinstance(exc, ValueError) else 500, {'error': str(exc)})
            finally:
                cancel.set()
                if stream is not None:
                    stream.close()
                app.release()

        def send_event(self, event):
            self.wfile.write((json.dumps(event, ensure_ascii=False) + '\n').encode())
            self.wfile.flush()

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    from .settings import load_settings
    settings = load_settings()
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(settings, args)
    config = json.loads((checkpoint / 'config.json').read_text())
    from .inference import load_model, file_hash, write_json
    from .distillation import student_from_checkpoint
    from .training_runs import new_session_id
    import subprocess

    print(f'Loading checkpoint: {checkpoint}', flush=True)
    tokenizer, source = load_model(settings, args.device)
    model = student_from_checkpoint(source, checkpoint, model_id=settings.model_id,
                                    model_revision=settings.model_revision, chunk_size=config['chunk_size'])
    del source
    app = ChatApplication(ChatEngine(model, tokenizer, args), args, checkpoint)
    server = make_server(app, args.host, args.port)
    record = settings.paths['LOG_DIR'] / f'chat-{new_session_id()}.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    write_json(record, {'checkpoint': str(checkpoint), 'adapter_sha256': file_hash(checkpoint / 'adapter.pt'),
                       'checkpoint_config': config, 'arguments': vars(args),
                       'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                       'source_dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()),
                       'lock_sha256': file_hash(root / 'uv.lock')})
    print(f'Startup record: {record}', flush=True)
    print(f'Listening on http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()


if __name__ == '__main__':
    main()
