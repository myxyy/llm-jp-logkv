"""Local chat server with one model replica per device. Settings precede HF imports.

POST /api/chat streams newline-delimited JSON. Each browser session retains its
LogKV state, including analysis, until reset or idle expiry. Only new input is
prefilled. Sessions stay on their assigned device; devices serve concurrently.
"""
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
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


def parse_devices(value):
    devices = []
    for part in value.split(','):
        part = part.strip()
        if part.isdecimal():
            part = f'cuda:{int(part)}'
        elif part == 'cuda':
            part = 'cuda:0'
        elif part.startswith('cuda:') and part[5:].isdecimal():
            part = f'cuda:{int(part[5:])}'
        elif part != 'cpu':
            raise ValueError('device must be cpu, cuda:N, or comma-separated GPU indices (e.g. 0,1,2)')
        if part in devices:
            raise ValueError('device indices must be distinct')
        devices.append(part)
    if len(devices) > 1 and 'cpu' in devices:
        raise ValueError('cpu cannot be combined with GPU devices')
    return devices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', help='Select a run under CHECKPOINT_DIR')
    parser.add_argument('--checkpoint', default='latest',
                        help='latest (requires --run-name), step name, or checkpoint directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda:0',
                        help='cpu, cuda:0, or GPU list such as 0,1,2 (one model per GPU)')
    parser.add_argument('--max-new-tokens', type=positive_int, default=2048,
                        help='Default and maximum output length per turn, including analysis')
    parser.add_argument('--session-ttl', type=positive_int, default=1800,
                        help='Seconds to retain idle session states (default: 1800)')
    parser.add_argument('--prefill-chunk-size', type=positive_int, default=128)
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], default='low')
    parser.add_argument('--temperature', type=float, default=0., help='0 = greedy; range 0–2')
    parser.add_argument('--top-p', type=float, default=1., help='Range (0, 1]')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error('temperature must be in [0, 2] and top-p in (0, 1]')
    try:
        args.devices = parse_devices(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    args.device = ','.join(args.devices)
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
    session_id(body)
    if type(body.get('turn')) is not int or body['turn'] < 0:
        raise ValueError('turn must be a nonnegative integer')
    prompt = body.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be a nonempty string')
    if 'messages' in body:
        raise ValueError('Send only the new prompt; conversation state is held by the server')
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
    return request_id, prompt, dict(options, max_new_tokens=limit, reasoning_effort=effort)


def session_id(body):
    value = body.get('session_id') if isinstance(body, dict) else None
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError('session_id is required')
    return value


@dataclass
class ChatSession:
    state: object = None
    effort: str | None = None
    turn: int = 0
    last_used: float = 0.
    worker: int | None = None


@dataclass
class PreparedTurn:
    ids: object
    initial_state: object
    result_state: object = None
    retained: bool = False


class SessionError(ValueError):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


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
        self.message_open = True  # The generation prompt ends in <|start|>assistant.

    def closing_tokens(self):
        if not self.message_open:
            return []
        names = ['end'] if self.in_body else ['message', 'end']
        return [self.tokenizer.convert_tokens_to_ids(f'<|{name}|>') for name in names]

    def push(self, token):
        self.tokens.append(token)
        kind = self.special.get(token)
        if kind in ('start', 'end', 'return'):
            self.message_open = kind == 'start'
            self.in_body, self.has_channel = False, False
            self.header = []
        elif kind == 'channel':
            self.in_body, self.has_channel = False, True
            self.header = []
        elif kind == 'message':
            self.message_open = True
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

    def prepare(self, prompt, options, session):
        if session.effort is not None and options['reasoning_effort'] != session.effort:
            raise ValueError('Reasoning effort is fixed for this session. Start a new conversation to change it.')
        rendered = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
            reasoning_effort=options['reasoning_effort'], conversation_start_date=self.conversation_date)
        if session.state is not None:
            # The pinned llm-jp Harmony template emits the system prefix once.
            # Past analysis/final tokens already live in the compressed state.
            marker = '<|start|>user<|message|>'
            _, separator, suffix = rendered.partition(marker)
            if not separator:
                raise ValueError('Expected llm-jp Harmony user framing in the chat template')
            rendered = separator + suffix
        ids = self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt').input_ids
        return PreparedTurn(ids.to(self.args.device), session.state)

    def stream(self, turn, options, cancel):
        import torch
        # inference_mode must be entered in the HTTP worker thread.
        device = torch.cuda.device(self.args.device) if self.args.device.startswith('cuda:') else nullcontext()
        with device, torch.inference_mode():
            ids, state = turn.ids, turn.initial_state
            cached = state.offset if state is not None else 0
            reason, count, prefilled = 'length', 0, 0
            eos = self.model.config.eos_token_id
            eos = set([eos] if isinstance(eos, int) else (eos or []))
            output = HarmonyOutput(self.tokenizer, eos)
            started = time.perf_counter()
            decode_started = None
            yield {'type': 'start', 'input_tokens': ids.shape[1], 'cached_tokens': cached}
            for part in ids.split(self.args.prefill_chunk_size, dim=1):
                if cancel.is_set():
                    break
                logits, state = self.model(part, state, logits_to_keep=1)
                prefilled += part.shape[1]
                yield {'type': 'prefill', 'processed_tokens': prefilled, 'input_tokens': ids.shape[1]}
            if not cancel.is_set():
                decode_started = time.perf_counter()
                for _ in range(options['max_new_tokens']):
                    if cancel.is_set():
                        break
                    token = sample_token(logits[0, -1], options['temperature'], options['top_p'])
                    count += 1
                    # In the original template, past assistant turns end in
                    # <|end|>; <|return|>/EOS is an output-only boundary. Consume
                    # that boundary as <|end|>, keeping all analysis/final bodies.
                    consume = output.closing_tokens() if token in eos else [token]
                    output.push(token)
                    if consume:
                        logits, state = self.model(ids.new_tensor([consume]), state, logits_to_keep=1)
                    yield {'type': 'update', 'output_tokens': count, **output.snapshot()}
                    if token in eos:
                        reason = 'eos'
                        break
            if cancel.is_set():
                reason = 'cancelled'
            # Only a fully prefilled user turn may be retained. Stop/length can
            # retain a partial assistant response, closed before the next user.
            # A disconnect/exception closes this generator without committing.
            if prefilled == ids.shape[1]:
                closing = output.closing_tokens()
                if closing:
                    _, state = self.model(ids.new_tensor([closing]), state, logits_to_keep=1)
                turn.result_state, turn.retained = state, True
            ended = time.perf_counter()
            snapshot = output.snapshot(finished=True)
            final = ''.join(p['text'] for p in snapshot['parts'] if p['channel'] == 'final')
            yield {'type': 'done', 'stop_reason': reason, 'input_tokens': ids.shape[1],
                   'cached_tokens': cached, 'session_tokens': state.offset if turn.retained else cached,
                   'retained': turn.retained, 'output_tokens': count, 'seconds': ended - started,
                   'tokens_per_second': count / max(ended - decode_started, 1e-9) if decode_started else 0,
                   'final': final, 'complete': reason == 'eos' and bool(final.strip()), **snapshot}


class ChatApplication:
    def __init__(self, engine, args, checkpoint, *, clock=time.monotonic):
        self.engines = list(engine) if isinstance(engine, (list, tuple)) else [engine]
        if not self.engines:
            raise ValueError('At least one chat engine is required')
        self.engine, self.args = self.engines[0], args
        self.checkpoint = str(checkpoint)
        self.lock = threading.Lock()
        self.active = {}
        self.sessions = {}
        self.clock = clock

    def _expire(self):
        now = self.clock()
        active_ids = {active[0] for active in self.active.values()}
        for key, session in list(self.sessions.items()):
            if key not in active_ids and now - session.last_used >= self.args.session_ttl:
                del self.sessions[key]

    def expire(self):
        with self.lock:
            self._expire()

    def create(self):
        with self.lock:
            self._expire()
            key = secrets.token_urlsafe(32)
            self.sessions[key] = ChatSession(last_used=self.clock())
            return key

    def discard(self, key):
        with self.lock:
            for active in self.active.values():
                if active[0] == key:
                    active[2].set()
            self.sessions.pop(key, None)

    def claim(self, key, expected_turn, request_id):
        with self.lock:
            self._expire()
            session = self.sessions.get(key)
            if session is None:
                raise SessionError(410, 'session_expired', 'Session expired or server restarted. Start a new conversation.')
            if any(active[0] == key for active in self.active.values()):
                raise SessionError(409, 'busy', 'Another response is being generated. Retry after it finishes.')
            if session.turn != expected_turn:
                raise SessionError(409, 'stale_turn', 'Session has advanced. Start a new conversation to avoid replaying a turn.')
            if session.worker is None:
                available = [i for i in range(len(self.engines)) if i not in self.active]
                if not available:
                    raise SessionError(409, 'busy', 'All devices are busy. Retry after a response finishes.')
                # Prefer the idle device retaining the fewest sessions; stable ties
                # follow CLI order. State is never moved between devices.
                session.worker = min(available, key=lambda i: sum(s.worker == i for s in self.sessions.values()))
            elif session.worker in self.active:
                raise SessionError(409, 'busy', 'This session\'s device is busy. Retry after it finishes.')
            cancel = threading.Event()
            self.active[session.worker] = (key, request_id, cancel)
            return session, cancel

    def commit(self, key, turn, effort):
        with self.lock:
            if key in self.sessions and turn.retained:
                session = self.sessions[key]
                session.state, session.effort = turn.result_state, effort
                session.turn += 1

    def stop(self, request_id=None, key=None):
        with self.lock:
            stopped = False
            for active in self.active.values():
                if request_id is None or (request_id == active[1] and key == active[0]):
                    active[2].set()
                    stopped = True
            return stopped

    def release(self, key, request_id):
        with self.lock:
            for worker, active in list(self.active.items()):
                if (key, request_id) != active[:2]:
                    continue
                if key in self.sessions:
                    session = self.sessions[key]
                    session.last_used = self.clock()
                    if session.state is None:
                        session.worker = None  # Failed/cancelled initial prefill.
                del self.active[worker]


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
                self.respond(200, {'checkpoint': app.checkpoint,
                    'devices': [engine.args.device for engine in app.engines], **{
                    key: getattr(app.args, key) for key in
                    ('max_new_tokens', 'session_ttl', 'temperature', 'top_p', 'reasoning_effort')}})
            else:
                self.respond(404, {'error': 'Not found'})

        def read_body(self):
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                raise ValueError('Cross-origin requests are not supported')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('Content-Type must be application/json')
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0:
                raise ValueError('A nonempty JSON body with Content-Length is required')
            return json.loads(self.rfile.read(size))

        def do_POST(self):
            if self.path not in ('/api/chat', '/api/stop', '/api/session', '/api/close'):
                self.respond(404, {'error': 'Not found'})
                return
            try:
                body = self.read_body()
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object')
                if self.path == '/api/session':
                    self.respond(200, {'session_id': app.create(), 'turn': 0})
                    return
                key = session_id(body)
                if self.path == '/api/close':
                    app.discard(key)
                    self.respond(200, {'closed': True})
                    return
                if self.path == '/api/stop':
                    if not isinstance(body, dict) or not isinstance(body.get('request_id'), str):
                        raise ValueError('request_id is required')
                    self.respond(200, {'stopped': app.stop(body['request_id'], key)})
                    return
                request_id, prompt, options = validate_request(body, app.args)
            except (ValueError, OSError) as exc:
                self.respond(400, {'error': str(exc)})
                return
            try:
                session, cancel = app.claim(key, body['turn'], request_id)
            except SessionError as exc:
                self.respond(exc.status, {'error': str(exc), 'code': exc.code})
                return
            stream = None
            streaming = False
            try:
                engine = app.engines[session.worker]
                turn = engine.prepare(prompt, options, session)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                streaming = True
                stream = engine.stream(turn, options, cancel)
                for event in stream:
                    if event['type'] == 'start':
                        event['device'] = engine.args.device
                    if event['type'] == 'done':
                        event['turn'] = session.turn + int(turn.retained)
                    self.send_event(event)
                    if event['type'] == 'done':
                        app.commit(key, turn, options['reasoning_effort'])
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
                try:
                    if stream is not None:
                        stream.close()
                finally:
                    app.release(key, request_id)

        def send_event(self, event):
            self.wfile.write((json.dumps(event, ensure_ascii=False) + '\n').encode())
            self.wfile.flush()

    class Server(ThreadingHTTPServer):
        def service_actions(self):
            # Run even without incoming requests; never expire an active turn.
            app.expire()

    return Server((host, port), Handler)


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
    import torch
    for device in args.devices:
        if device.startswith('cuda:') and int(device[5:]) >= torch.cuda.device_count():
            raise ValueError(f'Unavailable device: {device}; indices refer to CUDA_VISIBLE_DEVICES')
    engines = []
    for device in args.devices:
        print(f'Loading replica on {device}', flush=True)
        with torch.cuda.device(device) if device.startswith('cuda:') else nullcontext():
            tokenizer, source = load_model(settings, device)
            model = student_from_checkpoint(source, checkpoint, model_id=settings.model_id,
                                            model_revision=settings.model_revision, chunk_size=config['chunk_size'])
            del source
        worker_args = argparse.Namespace(**{**vars(args), 'device': device})
        engines.append(ChatEngine(model, tokenizer, worker_args))
    app = ChatApplication(engines, args, checkpoint)
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
