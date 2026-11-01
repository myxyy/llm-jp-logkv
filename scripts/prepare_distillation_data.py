#!/usr/bin/env python3
"""Download pinned reasoning_low shards, then build a new uint32 cache."""
import argparse
from pathlib import Path

from llm_jp_logkv.settings import load_settings
settings = load_settings()

from llm_jp_logkv.training_data import DATASET_ID, DATASET_REVISION, SUBSETS, prepare_cache

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--download', action='store_true')
parser.add_argument('--max-length', type=int, default=2049)
parser.add_argument('--candidates-per-subset', type=int, default=30000)
parser.add_argument('--seed', type=int, default=20261102)
args = parser.parse_args()
if not args.output.resolve().is_relative_to(settings.paths['DATASET_DIR'].resolve()):
    parser.error('output must be under configured DATASET_DIR')
if args.output.exists():
    parser.error('output already exists; choose a new cache directory')
if args.download:
    from huggingface_hub import snapshot_download
    snapshot_download(DATASET_ID, repo_type='dataset', revision=DATASET_REVISION,
        local_dir=settings.paths['DATASET_DIR']/'llm-jp-4.1-thinking-sft-data'/DATASET_REVISION,
        allow_patterns=['README.md', *[f'{subset}/reasoning_low-*.parquet' for subset in SUBSETS]])
manifest = prepare_cache(settings, args.output, max_length=args.max_length,
                         candidates_per_subset=args.candidates_per_subset, seed=args.seed)
print('Cache:', args.output)
print('Train/validation examples:', manifest['train_examples'], manifest['valid_examples'])
