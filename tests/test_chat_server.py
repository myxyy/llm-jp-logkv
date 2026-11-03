"""Streaming, Harmony framing, cancellation, and the local HTTP contract."""
from llm_jp_logkv.settings import load_settings
load_settings()

import http.client
import json
import socket
import struct
from types import SimpleNamespace
import threading
import time

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_jp_logkv.chat_server import (ChatApplication, ChatEngine, HarmonyOutput,
    make_server, parse_args, resolve_checkpoint, sample_token, validate_request)
from llm_jp_logkv.llama_model import LlamaLogKVForCausalLM


class Tokenizer:
    special = {'<|start|>': 1, '<|channel|>': 2, '<|message|>': 3,
               '<|end|>': 4, '<|return|>': 5}

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def decode(self, tokens, **kwargs):
        self.last_decoded = [int(token) for token in tokens]
        vocabulary = {value: key.encode() for key, value in self.special.items()}
        vocabulary.update({6: b'analysis', 7: b'final', 8: b'assistant',
                           9: b'\xe6', 10: b'\x97', 11: b'\xa5', 12: b'42'})
        return b''.join(vocabulary.get(int(token), b'x') for token in tokens).decode(errors='replace')

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return SimpleNamespace(input_ids=torch.tensor([[15, 16, 17]]))


def test_harmony_channels_utf8_and_raw_partial_headers():
    output = HarmonyOutput(Tokenizer())
    for token in [2, 6, 3, 9]:
        output.push(token)
    assert output.snapshot()['parts'] == [{'channel': 'analysis', 'text': ''}]
    for token in [10, 11, 4, 1, 8, 2, 7]:
        output.push(token)
    assert output.snapshot()['parts'] == [{'channel': 'analysis', 'text': '日'}]
    assert output.snapshot()['raw'].endswith('<|channel|>final')
    for token in [3, 12, 5]:
        output.push(token)
    assert output.snapshot()['parts'][-1] == {'channel': 'final', 'text': '42'}
    assert output.snapshot()['raw'].endswith('42<|return|>')


class ScriptedModel:
    config = SimpleNamespace(eos_token_id=[5, 19], max_position_embeddings=128)
    sequence = [2, 6, 3, 9, 10, 11, 4, 1, 8, 2, 7, 3, 12, 5]

    def __init__(self):
        self.calls = []

    def __call__(self, ids, state=None, **kwargs):
        assert torch.is_inference_mode_enabled()
        offset = (state.offset if state else 0) + ids.shape[1]
        self.calls.append((ids.tolist(), offset))
        logits = torch.full((1, 1, 64), -100.)
        logits[0, 0, self.sequence[max(0, offset - 3)]] = 100.
        return logits, SimpleNamespace(offset=offset)


def engine(model=None):
    args = parse_args(['--device', 'cpu', '--prefill-chunk-size', '2', '--max-new-tokens', '20'])
    return ChatEngine(model or ScriptedModel(), Tokenizer(), args)


def request(**kwargs):
    return {'request_id': 'request-1', 'messages': [{'role': 'user', 'content': 'こんにちは'}], **kwargs}


def test_stream_eos_channel_transition_counts_and_multiturn_template():
    chat = engine()
    messages = [{'role': 'user', 'content': '前の質問'}, {'role': 'assistant', 'content': '前の回答'},
                {'role': 'user', 'content': '次の質問'}]
    _, messages, options = validate_request(request(messages=messages, reasoning_effort='high'), chat.args)
    ids = chat.prepare(messages, options)
    events = list(chat.stream(ids, options, threading.Event()))
    assert chat.tokenizer.messages == messages
    assert chat.tokenizer.kwargs['reasoning_effort'] == 'high'
    assert chat.tokenizer.kwargs['add_generation_prompt']
    assert [e['processed_tokens'] for e in events if e['type'] == 'prefill'] == [2, 3]
    done = events[-1]
    assert done['complete'] and done['final'] == '42' and done['stop_reason'] == 'eos'
    assert done['input_tokens'] == 3 and done['output_tokens'] == len(chat.model.sequence)
    assert done['tokens_per_second'] > 0
    assert done['parts'][0] == {'channel': 'analysis', 'text': '日'}
    # EOS is not fed back: state is deliberately discarded at the turn boundary.
    assert chat.model.calls[-1][0] == [[12]]


@pytest.mark.parametrize('phase', ['start', 'prefill', 'update'])
def test_cancel_during_prefill_or_decode_does_not_start_more_model_calls(phase):
    chat = engine()
    _, messages, options = validate_request(request(), chat.args)
    cancel = threading.Event()
    events = chat.stream(chat.prepare(messages, options), options, cancel)
    for event in events:
        if event['type'] == phase:
            cancel.set()
            calls = len(chat.model.calls)
            break
    tail = list(events)
    assert tail[-1]['stop_reason'] == 'cancelled' and not tail[-1]['complete']
    assert len(chat.model.calls) == calls


def test_limit_during_analysis_is_not_a_completed_answer():
    chat = engine()
    _, messages, options = validate_request(request(max_new_tokens=6), chat.args)
    done = list(chat.stream(chat.prepare(messages, options), options, threading.Event()))[-1]
    assert done['stop_reason'] == 'length' and not done['complete'] and not done['final']
    assert done['parts'] == [{'channel': 'analysis', 'text': '日'}]


def test_alternate_model_eos_is_not_part_of_answer_text():
    model = ScriptedModel()
    model.sequence = [*model.sequence[:-1], 19]
    chat = engine(model)
    _, messages, options = validate_request(request(), chat.args)
    done = list(chat.stream(chat.prepare(messages, options), options, threading.Event()))[-1]
    assert done['complete'] and done['final'] == '42'


def test_greedy_stream_matches_existing_logkv_generation():
    torch.manual_seed(29)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=128, eos_token_id=[])
    model = LlamaLogKVForCausalLM(LlamaForCausalLM(config).eval())
    chat = engine(model)
    _, messages, options = validate_request(request(max_new_tokens=12), chat.args)
    ids = chat.prepare(messages, options)
    tokens, _, _ = model.greedy_generate(ids, max_new_tokens=12, prefill_chunk_size=2)
    done = list(chat.stream(ids, options, threading.Event()))[-1]
    assert chat.tokenizer.last_decoded == tokens[0].tolist()
    assert done['raw'] == chat.tokenizer.decode(tokens[0])
    assert done['output_tokens'] == 12


def test_sampling_top_p_and_nonfinite_logits():
    logits = torch.tensor([0., 10., 1.])
    assert sample_token(logits, 0, 1) == 1
    assert all(sample_token(logits, 1, .01) == 1 for _ in range(10))
    assert sample_token(logits, 1, 1) in (0, 1, 2)
    with pytest.raises(FloatingPointError):
        sample_token(torch.tensor([float('nan')]), 0, 1)


@pytest.mark.parametrize('change', [dict(temperature=float('nan')), dict(top_p=0), dict(top_p=True),
    dict(max_new_tokens=0), dict(max_new_tokens=21), dict(max_new_tokens=1.5),
    dict(reasoning_effort='invalid'), dict(messages=[]), dict(messages=[{'role': 'assistant', 'content': 'x'}]),
    dict(messages=[{'role': 'user', 'content': ' '}]), dict(request_id=42)])
def test_invalid_requests(change):
    with pytest.raises(ValueError):
        validate_request(request(**change), engine().args)


def test_input_and_context_limits():
    chat = engine()
    _, messages, options = validate_request(request(), chat.args)
    chat.args.max_input_tokens = 2
    with pytest.raises(ValueError, match='Input has 3 tokens'):
        chat.prepare(messages, options)
    chat.args.max_input_tokens = 100
    options['max_new_tokens'] = 128
    with pytest.raises(ValueError, match='context length'):
        chat.prepare(messages, options)


def test_checkpoint_selection(tmp_path):
    settings = SimpleNamespace(paths={'CHECKPOINT_DIR': tmp_path})
    root = tmp_path / 'test'
    for step in (9, 10, 11):
        path = root / f'step-{step}'
        path.mkdir(parents=True)
        for name in ['adapter.pt', 'training_state.pt', 'config.json']:
            (path / name).write_text('{}')
        if step != 11:
            (path / 'complete.json').write_text(json.dumps({'step': step, 'source_run': '/run'}))
    args = parse_args(['--run-name', 'test'])
    assert resolve_checkpoint(settings, args) == root / 'step-10'
    args.checkpoint = 'step-9'
    assert resolve_checkpoint(settings, args) == root / 'step-9'
    args.run_name = None
    args.checkpoint = str(root / 'step-10')
    assert resolve_checkpoint(settings, args) == root / 'step-10'
    args.checkpoint = 'latest'
    with pytest.raises(ValueError, match='requires --run-name'):
        resolve_checkpoint(settings, args)
    args.run_name = 'missing'
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(settings, args)


@pytest.fixture
def http_server():
    chat = engine()
    app = ChatApplication(chat, chat.args, '/checkpoint/step-1')
    server = make_server(app, '127.0.0.1', 0)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        yield app, server.server_port
    finally:
        app.stop()
        server.shutdown()
        server.server_close()
        worker.join()


def post(port, path, body, **headers):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    connection.request('POST', path, json.dumps(body), {'Content-Type': 'application/json', **headers})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


def test_http_stream_errors_and_lock_release(http_server):
    app, port = http_server
    for path, content in [('/', b'<title>llm-jp LogKV Chat</title>'),
                          ('/api/config', b'/checkpoint/step-1')]:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        connection.request('GET', path)
        response = connection.getresponse()
        assert response.status == 200 and content in response.read()
        connection.close()
    status, data = post(port, '/api/chat', request())
    events = [json.loads(line) for line in data.splitlines()]
    assert status == 200 and events[0]['type'] == 'start' and events[-1]['complete']
    assert app.active is None
    assert post(port, '/api/chat', request(top_p=0))[0] == 400
    assert post(port, '/api/chat', request(), Origin='http://elsewhere')[0] == 400
    assert post(port, '/api/stop', {})[0] == 400
    # Errors after streaming starts are explicit, and release the model lease.
    original = app.engine.model
    app.engine.model = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('test failure'))
    app.engine.model.config = original.config
    status, data = post(port, '/api/chat', request())
    assert status == 200 and json.loads(data.splitlines()[-1])['type'] == 'error'
    assert app.active is None
    app.engine.model = original
    assert post(port, '/api/chat', request())[0] == 200


def test_http_stop_and_busy_while_stream_is_open(http_server):
    app, port = http_server
    entered = threading.Event()
    def waiting_stream(ids, options, cancel):
        yield {'type': 'start', 'input_tokens': 3}
        entered.set()
        assert cancel.wait(5)
        yield {'type': 'done', 'stop_reason': 'cancelled'}
    app.engine.stream = waiting_stream
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    connection.request('POST', '/api/chat', json.dumps(request()), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    assert json.loads(response.readline())['type'] == 'start'
    assert entered.wait(5)
    assert post(port, '/api/chat', request(request_id='other'))[0] == 409
    assert json.loads(post(port, '/api/stop', {'request_id': 'other'})[1]) == {'stopped': False}
    assert json.loads(post(port, '/api/stop', {'request_id': 'request-1'})[1]) == {'stopped': True}
    assert json.loads(response.read())['stop_reason'] == 'cancelled'
    connection.close()
    assert app.active is None


def test_disconnect_cancels_generator_and_releases_model(http_server):
    app, port = http_server
    disconnected, closed = threading.Event(), threading.Event()
    def disconnect_stream(ids, options, cancel):
        try:
            yield {'type': 'start'}
            assert disconnected.wait(5)
            for _ in range(10):
                yield {'type': 'update', 'raw': 'x' * 100_000}
        finally:
            assert cancel.is_set()
            closed.set()
    app.engine.stream = disconnect_stream
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    connection.connect()
    transport = connection.sock
    connection.request('POST', '/api/chat', json.dumps(request()), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    assert json.loads(response.readline())['type'] == 'start'
    transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
    response.close()
    connection.close()
    disconnected.set()
    assert closed.wait(5)
    # Generator cleanup precedes release of the lease in the handler's finally.
    deadline = time.monotonic() + 5
    while app.active is not None and time.monotonic() < deadline:
        time.sleep(.01)
    assert app.active is None
