"""Named training runs, completed checkpoints and per-launch records."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4


def validate_run_name(name):
    if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('--run-name must be a single directory name')
    return name


def named_directory(settings, key, name):
    validate_run_name(name)
    root = settings.paths[key].resolve()
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f'Run directory must stay within {root}: {path}')
    return path


def checkpoint_metadata(path):
    number = path.name.removeprefix('step-')
    if (not path.name.startswith('step-') or not number.isascii() or not number.isdecimal()
            or path.is_symlink() or not path.is_dir()):
        return None
    if not all((path/name).is_file() for name in ('adapter.pt', 'training_state.pt', 'config.json', 'complete.json')):
        return None
    try:
        metadata = json.loads((path/'complete.json').read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(metadata, dict) or metadata.get('step') != int(number)
            or not isinstance(metadata.get('source_run'), str)):
        return None
    return metadata


def completed_checkpoints(root, source_run=None):
    found = []
    for path in root.glob('step-*'):
        metadata = checkpoint_metadata(path)
        if metadata is not None and (source_run is None
                or Path(metadata['source_run']).resolve() == source_run.resolve()):
            found.append(path)
    return sorted(found, key=lambda path: int(path.name.removeprefix('step-')))


def resolve_resume(args, settings):
    """Run under the run lock; never silently fork/rewind an existing run."""
    root = named_directory(settings, 'CHECKPOINT_DIR', args.run_name)
    existing = completed_checkpoints(root)
    resume = None
    if args.resume:
        if str(args.resume) == 'latest':
            if not existing:
                raise FileNotFoundError(f'No completed checkpoints in {root}')
            resume = existing[-1]
        else:
            resume = Path(args.resume).expanduser()
            if not resume.is_absolute() and resume.parent == Path('.'):
                resume = root/resume
        resume = resume.resolve()
        if checkpoint_metadata(resume) is None:
            raise ValueError(f'Not a completed checkpoint: {resume}')
    if existing:
        if resume is None:
            raise FileExistsError(f'Run {args.run_name} already has checkpoints; use --resume latest or a new --run-name')
        if resume != existing[-1].resolve():
            raise ValueError('Output run has other/later checkpoints; use --resume latest or a new --run-name')
        if Path(checkpoint_metadata(resume)['source_run']).resolve() != args.run_dir.resolve():
            raise ValueError('Output checkpoint directory belongs to a different run')
    elif resume is None and ((args.run_dir/'manifest.json').exists()
                            or any((args.run_dir/'sessions').glob('*'))):
        raise FileExistsError('Run already has launch records; use a new --run-name or --resume CHECKPOINT')
    return resume


def new_session_id():
    return f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}'


def create_session(root, session_id):
    validate_run_name(session_id)
    path = root/'sessions'/session_id
    path.mkdir(parents=True, exist_ok=False)
    latest = root/'latest'
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError(f'Expected a session symlink: {latest}')
    temporary = root/f'.latest-{session_id}'
    temporary.symlink_to(Path('sessions')/session_id, target_is_directory=True)
    os.replace(temporary, latest)
    return path


@contextmanager
def exclusive_run(run_dir, inherited_token=None):
    """Supervisor holds the lock through evaluation; its torchrun inherits the FD.

    Rank zero may join that same lease using the private token passed in its
    environment. Direct torchrun launches instead acquire their own lock.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir/'.training.lock').open('a+') as lock:
        token = inherited_token
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            if not inherited_token or lock.read().strip() != inherited_token:
                raise RuntimeError(f'Another trainer is using {run_dir}') from None
        else:
            token = inherited_token or uuid4().hex
            lock.seek(0)
            lock.truncate()
            lock.write(token+'\n')
            lock.flush()
        # Closing the descriptor releases the lock. An inherited copy in
        # torchrun keeps it held if the supervisor is killed first.
        yield token, lock.fileno()
