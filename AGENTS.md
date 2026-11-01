# Project scope

- Work in `/home/myxy/llm-jp-logkv`. This is a separate repository for investigating
  pretrained `llm-jp/llm-jp-4.1-8b-thinking` attention conversion to LogKV.
- Read `README.md`, `doc/handoff-20261101.md`, and `UPSTREAM.json` before architectural work.
- `/home/myxy/RecursiveCompressor` is the upstream reference. Changes there require
  a task that includes that repository; runtime must not depend on its working tree.
- Current status is scaffolding plus a copied LogKV core. Do not imply that Llama
  conversion, distillation, or preservation of reasoning quality is already implemented.

# Environment and artifacts

- Use uv: `./scripts/uv.sh sync --locked` and `./scripts/uv.sh run --locked ...`.
  The wrapper loads `.env` before uv chooses its cache and virtual environment.
- Keep model weights, datasets, hidden-state caches, checkpoints, logs, and Python
  package caches under the configured `/mnt/raid0/llm-jp-logkv` paths.
- Load settings before importing Hugging Face packages. Do not print `.env` or tokens.
- The project `.env` takes precedence over inherited Python/HF settings. Prefer the
  uv wrapper so this also holds when selecting uv's environment and package cache.
- Keep `.env` out of Git; update `.env.example` for new settings. Pin model revisions
  and dependency versions in recorded manifests and `uv.lock`.
- Do not overwrite earlier experiment artifacts. Record source/model revisions,
  settings, and validation results for each experiment.

# Implementation and verification

- Keep the imported LogKV core traceable to `UPSTREAM.json`; record any subsequent
  semantic changes. Build Llama-specific adapters separately from the reference core.
- Preserve original Llama tokenizer, chat template, RMSNorm semantics, and FFN when
  evaluating an attention-only change. GQA and RoPE compatibility are open design work.
- LogKV pooling is query-dependent: repeating GQA KV heads before pooling does not
  preserve shared higher-level states. Do not claim an 8-head cache without implementing it.
- Do not reuse the upstream uint16 data cache: llm-jp vocabulary is 196,608.
- Normal CPU verification:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./scripts/uv.sh run --locked pytest -q`.
- For attention changes preserve the independent FP64 reference and streaming equivalence;
  low-precision comparisons need explicit tolerances. Report quality and performance separately.
- CPU setup checks do not authorize a large training campaign. Carry forward the user's
  machine constraint: at most two GPUs for continuous experiments; ask before a proposed
  experiment expected to take eight hours or longer. Later explicit user instructions prevail.
