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

from llm_jp_logkv.chat_server import (ChatApplication, ChatEngine, ChatSession, HarmonyOutput, SessionError,
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
        return ('SYSTEM ' + kwargs['reasoning_effort'] + '<|start|>user<|message|>'
                + messages[0]['content'] + '<|end|><|start|>assistant')

    def __call__(self, text, **kwargs):
        self.encoded_text = text
        prompt = text.split('<|start|>user<|message|>')[1].split('<|end|>')[0]
        ids = ([20, 21] if text.startswith('SYSTEM') else [])
        ids += [1, 22, 3] + [32 + ord(c) % 16 for c in prompt] + [4, 1, 8]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


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
        previous = state.tokens if state else ()
        tokens = previous + tuple(ids[0].tolist())
        offset = len(tokens)
        self.calls.append((ids.tolist(), offset))
        # Find the current user and its following assistant prefix.
        current = tokens[len(tokens) - 1 - tokens[::-1].index(22):] if 22 in tokens else ()
        index = len(current) - current.index(8) - 1 if 8 in current else 0
        logits = torch.full((1, 1, 64), -100.)
        logits[0, 0, self.sequence[min(index, len(self.sequence) - 1)]] = 100.
        return logits, SimpleNamespace(offset=offset, tokens=tokens)


def engine(model=None):
    args = parse_args(['--device', 'cpu', '--prefill-chunk-size', '2', '--max-new-tokens', '20'])
    return ChatEngine(model or ScriptedModel(), Tokenizer(), args)


def request(**kwargs):
    return {'session_id': 'test-session', 'turn': 0, 'request_id': 'request-1', 'prompt': 'こんにちは', **kwargs}


def run_turn(chat, session=None, **kwargs):
    _, prompt, options = validate_request(request(**kwargs), chat.args)
    turn = chat.prepare(prompt, options, session or ChatSession())
    events = list(chat.stream(turn, options, threading.Event()))
    return turn, events


def test_stream_retains_analysis_final_and_only_prefills_new_turn():
    chat = engine()
    first, events = run_turn(chat, reasoning_effort='high')
    done = events[-1]
    assert chat.tokenizer.kwargs['reasoning_effort'] == 'high'
    assert done['complete'] and done['final'] == '42' and done['stop_reason'] == 'eos'
    assert done['input_tokens'] == first.ids.shape[1]
    assert done['output_tokens'] == len(chat.model.sequence)
    assert done['parts'][0] == {'channel': 'analysis', 'text': '日'}
    # Normalize only EOS to the template's past-turn delimiter; retain bodies.
    expected = tuple(first.ids[0].tolist() + chat.model.sequence[:-1] + [4])
    assert first.result_state.tokens == expected
    session = ChatSession(state=first.result_state, effort='high', turn=1)
    second, events = run_turn(chat, session, prompt='次の質問', reasoning_effort='high')
    assert chat.tokenizer.messages == [{'role': 'user', 'content': '次の質問'}]
    assert not chat.tokenizer.encoded_text.startswith('SYSTEM')
    assert events[0]['cached_tokens'] == len(expected)
    assert second.result_state.tokens == expected + tuple(second.ids[0].tolist() + chat.model.sequence[:-1] + [4])
    assert first.result_state.tokens == expected  # Prior state is not mutated.
    with pytest.raises(ValueError, match='fixed for this session'):
        run_turn(chat, session, reasoning_effort='low')


@pytest.mark.parametrize('phase', ['start', 'prefill', 'update'])
def test_cancel_rolls_back_incomplete_input_and_retains_partial_output(phase):
    chat = engine()
    _, prompt, options = validate_request(request(), chat.args)
    turn = chat.prepare(prompt, options, ChatSession())
    cancel = threading.Event()
    events = chat.stream(turn, options, cancel)
    for event in events:
        if event['type'] == phase:
            cancel.set()
            calls = len(chat.model.calls)
            break
    tail = list(events)
    assert tail[-1]['stop_reason'] == 'cancelled' and not tail[-1]['complete']
    if phase == 'update':
        assert tail[-1]['retained']
        assert turn.result_state.tokens[-2:] == (3, 4)  # Close a partial header.
        assert len(chat.model.calls) == calls + 1
    else:
        assert not tail[-1]['retained'] and turn.result_state is None
        assert len(chat.model.calls) == calls


def test_limit_during_analysis_keeps_thoughts_and_closes_turn():
    chat = engine()
    turn, events = run_turn(chat, max_new_tokens=6)
    done = events[-1]
    assert done['stop_reason'] == 'length' and not done['complete'] and not done['final']
    assert done['parts'] == [{'channel': 'analysis', 'text': '日'}]
    assert done['retained'] and turn.result_state.tokens[-4:] == (9, 10, 11, 4)


def test_alternate_model_eos_is_not_part_of_answer_text():
    model = ScriptedModel()
    model.sequence = [*model.sequence[:-1], 19]
    turn, events = run_turn(engine(model))
    assert events[-1]['complete'] and events[-1]['final'] == '42'
    assert turn.result_state.tokens[-2:] == (12, 4)


def tiny_model(**options):
    torch.manual_seed(29)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=8, eos_token_id=[5])
    return LlamaLogKVForCausalLM(LlamaForCausalLM(config).double().eval(), **options)


@pytest.mark.parametrize('options', [{}, {'no_rope': True, 'conv_kernel_size': 4}])
def test_cached_real_logkv_matches_continuous_tokens_beyond_original_context(monkeypatch, options):
    model = tiny_model(**options)
    chat = engine(model)
    sequence = ScriptedModel.sequence
    samples = iter(sequence * 2)
    monkeypatch.setattr('llm_jp_logkv.chat_server.sample_token', lambda *args: next(samples))
    first, _ = run_turn(chat)
    session = ChatSession(state=first.result_state, effort='low')
    second, _ = run_turn(chat, session, prompt='もう一度')
    all_ids = torch.cat([first.ids, torch.tensor([sequence[:-1] + [4]]),
                         second.ids, torch.tensor([sequence[:-1] + [4]])], dim=1)
    assert all_ids.shape[1] > model.config.max_position_embeddings
    # Independent replay from token zero checks both hidden contents and offsets.
    with torch.inference_mode():
        _, expected = model(all_ids, logits_to_keep=1)
        actual_logits, actual = model(torch.tensor([[32]]), second.result_state, logits_to_keep=1)
        expected_logits, _ = model(torch.tensor([[32]]), expected, logits_to_keep=1)
    assert actual.offset == all_ids.shape[1] + 1
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-10, rtol=1e-10)
    for a, b in zip(second.result_state.conv_caches, expected.conv_caches):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    for a, b in zip(second.result_state.layers, expected.layers):
        assert a.offset == b.offset
        for level_a, level_b in zip(a.payload[0], b.payload[0]):
            for tensor_a, tensor_b in zip(level_a, level_b):
                torch.testing.assert_close(tensor_a, tensor_b, atol=1e-10, rtol=1e-10)


def test_greedy_stream_matches_existing_logkv_generation():
    model = tiny_model()
    model.config.eos_token_id = []
    chat = engine(model)
    _, prompt, options = validate_request(request(max_new_tokens=12), chat.args)
    turn = chat.prepare(prompt, options, ChatSession())
    tokens, _, _ = model.greedy_generate(turn.ids, max_new_tokens=12, prefill_chunk_size=2)
    done = list(chat.stream(turn, options, threading.Event()))[-1]
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
    dict(reasoning_effort='invalid'), dict(prompt=''), dict(prompt=42), dict(session_id=None),
    dict(turn=True), dict(turn=-1), dict(messages=[]), dict(request_id=42)])
def test_invalid_requests(change):
    with pytest.raises(ValueError):
        validate_request(request(**change), engine().args)


def test_input_limit_option_removed_and_large_prompt_accepted():
    chat = engine()
    with pytest.raises(SystemExit):
        parse_args(['--max-input-tokens', '8192'])
    _, prompt, options = validate_request(request(prompt='長' * 9000), chat.args)
    turn = chat.prepare(prompt, options, ChatSession())
    assert turn.ids.shape[1] > 8192
    assert turn.ids.shape[1] > chat.model.config.max_position_embeddings


def test_sessions_isolated_expire_when_idle_and_cannot_replay_turns():
    chat = engine()
    now = [100.]
    app = ChatApplication(chat, chat.args, '/checkpoint', clock=lambda: now[0])
    one, two = app.create(), app.create()
    first, _ = run_turn(chat)
    session, cancel = app.claim(one, 0, 'r1')
    app.commit(one, first, 'low')
    assert session.state is first.result_state
    assert app.sessions[two].state is None
    now[0] += chat.args.session_ttl + 1
    app.expire()
    assert one in app.sessions and two not in app.sessions  # Active protected.
    app.release()
    with pytest.raises(SessionError, match='advanced'):
        app.claim(one, 0, 'replayed')
    app.claim(one, 1, 'r2')
    app.discard(one)
    assert cancel is not None and one not in app.sessions
    assert app.active[2].is_set()
    app.commit(one, first, 'low')  # Reset during generation must not resurrect it.
    app.release()
    assert one not in app.sessions
    three = app.create()
    now[0] += chat.args.session_ttl
    app.expire()
    with pytest.raises(SessionError) as error:
        app.claim(three, 0, 'expired')
    assert error.value.status == 410


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
    app.sessions['test-session'] = ChatSession(last_used=app.clock())
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
    status, data = post(port, '/api/chat', request(turn=1))
    assert status == 200 and json.loads(data.splitlines()[-1])['type'] == 'error'
    assert app.active is None
    app.engine.model = original
    assert post(port, '/api/chat', request(turn=1))[0] == 200


def test_http_sessions_continue_reset_and_expire_without_replaying_history(http_server):
    app, port = http_server
    status, data = post(port, '/api/session', {})
    session = json.loads(data)
    assert status == 200 and session['turn'] == 0
    key = session['session_id']
    status, data = post(port, '/api/chat', request(session_id=key))
    done = json.loads(data.splitlines()[-1])
    assert status == 200 and done['turn'] == 1 and done['retained']
    original = app.sessions[key].state
    assert app.sessions['test-session'].state is None
    assert post(port, '/api/chat', request(session_id=key))[0] == 409
    assert post(port, '/api/chat', request(session_id=key, turn=1, reasoning_effort='high'))[0] == 400
    assert app.sessions[key].state is original
    status, data = post(port, '/api/chat', request(session_id=key, turn=1, prompt='次', max_new_tokens=1))
    events = [json.loads(line) for line in data.splitlines()]
    assert status == 200 and events[0]['cached_tokens'] == original.offset
    assert events[-1]['turn'] == 2 and events[-1]['retained'] and not events[-1]['complete']
    assert app.sessions[key].state.tokens[:original.offset] == original.tokens
    assert post(port, '/api/close', {'session_id': key})[0] == 200
    assert key not in app.sessions
    assert post(port, '/api/chat', request(session_id=key, turn=2))[0] == 410
    # The server's periodic service action expires idle states without new calls.
    app.sessions['test-session'].last_used -= app.args.session_ttl + 1
    deadline = time.monotonic() + 3
    while 'test-session' in app.sessions and time.monotonic() < deadline:
        time.sleep(.01)
    assert 'test-session' not in app.sessions


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
    assert json.loads(post(port, '/api/stop', {'session_id': 'test-session', 'request_id': 'other'})[1]) == {'stopped': False}
    assert json.loads(post(port, '/api/stop', {'session_id': 'test-session', 'request_id': 'request-1'})[1]) == {'stopped': True}
    assert json.loads(response.read())['stop_reason'] == 'cancelled'
    connection.close()
    assert app.active is None


def test_disconnect_cancels_generator_and_releases_model(http_server):
    app, port = http_server
    previous, _ = run_turn(app.engine)
    app.sessions['test-session'].state = previous.result_state
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
    assert app.sessions['test-session'].state is previous.result_state
