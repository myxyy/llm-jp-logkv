"""Launch one teacher and student-only FSDP2 with the usual named-run supervisor."""
import argparse
import sys

from . import train_fsdp
from .training_launcher import launch


def parse_args(argv=None):
    parser = train_fsdp.build_parser()
    parser.add_argument('--gpus', default='1,2,3,4,5', help='Student GPU indices only')
    parser.add_argument('--teacher-gpu', default='0', help='Dedicated teacher GPU index')
    parser.add_argument('--evaluate-final', action='store_true')
    parser.add_argument('--evaluation-max-new-tokens', type=int, default=512)
    argv = [v for v in (sys.argv[1:] if argv is None else argv) if v != '--']
    args = train_fsdp.parse_args(argv, parser)
    gpus = args.gpus.split(',') + [args.teacher_gpu]
    if any(not g.isdecimal() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('Student and teacher GPUs must be distinct numeric indices')
    if args.evaluation_max_new_tokens < 1:
        parser.error('--evaluation-max-new-tokens must be positive')
    return args


def main():
    args = parse_args()
    combined = argparse.Namespace(**vars(args))
    combined.gpus = args.gpus + ',' + args.teacher_gpu
    try:
        code = launch(combined, trainer=train_fsdp, module='llm_jp_logkv.train_fsdp')
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(code)
