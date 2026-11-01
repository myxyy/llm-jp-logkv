"""Storage, pinned model download, and no-distillation LogKV inference."""

import argparse
from datetime import datetime, timezone
from uuid import uuid4
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from .settings import Settings, load_settings


METADATA_FILES = [
    "README.md", "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "chat_template.jinja", "LICENSE", "LICENSE.txt",
]


def doctor(settings: Settings) -> dict:
    import torch

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {name: version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "datasets", "huggingface-hub",
        )},
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "paths": {name: {"path": str(path), "exists": path.is_dir(),
                         "writable": os.access(path, os.W_OK)}
                  for name, path in settings.paths.items()},
        "logkv_source": str(settings.source_dir),
        "base_model": {"id": settings.model_id, "revision": settings.model_revision,
                       "path": str(settings.model_path),
                       "config_present": (settings.model_path / "config.json").is_file(),
                       "weight_files": len(list(settings.model_path.glob("*.safetensors")))},
    }


def download_model(settings: Settings, *, weights: bool) -> dict:
    # Import only after load_settings has configured all HF cache paths.
    from huggingface_hub import snapshot_download

    patterns = METADATA_FILES.copy()
    if weights:
        patterns += ["*.safetensors", "model.safetensors.index.json"]
    snapshot_download(
        settings.model_id,
        revision=settings.model_revision,
        local_dir=settings.model_path,
        cache_dir=settings.paths["HF_HUB_CACHE"],
        allow_patterns=patterns,
    )
    manifest = {"model_id": settings.model_id, "revision": settings.model_revision,
                "weights_requested": weights, "allow_patterns": patterns}
    manifest["recorded_at"] = datetime.now(timezone.utc).isoformat()
    record_name = f"download_manifest-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.json"
    (settings.model_path / record_name).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "path": str(settings.model_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Local .env file (default: cwd/.env)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-storage", help="Create configured storage directories")
    commands.add_parser("doctor", help="Report versions and storage; no model is loaded")
    download = commands.add_parser("download-model", help="Download pinned config and tokenizer")
    download.add_argument("--weights", action="store_true", help="Also download the full BF16 weights")
    for name, help_text in (("infer", "Generate from local llm-jp weights"),
                            ("evaluate-initial", "Compare original and untrained LogKV")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--chunk-size", type=int, default=4)
        command.add_argument("--prefill-chunk-size", type=int, default=128)
        command.add_argument("--max-new-tokens", type=int, default=512)
        command.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
        command.add_argument("--conversation-start-date", default="2026-11-01")
        if name == "infer":
            command.add_argument("--prompt", required=True)
            command.add_argument("--mode", choices=["original", "logkv"], default="logkv")
            command.add_argument("--no-level-bias", action="store_true")
        else:
            command.add_argument("--suite", type=Path, default=Path("evaluation/initial-quality.json"))
    args = parser.parse_args()
    if args.command in ("infer", "evaluate-initial"):
        if args.chunk_size < 2 or args.prefill_chunk_size < 1 or args.max_new_tokens < 1:
            parser.error("chunk-size >= 2 and positive token limits are required")
    try:
        settings = load_settings(args.env_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "prepare-storage":
        settings.prepare_storage()
        result = {name: str(path) for name, path in settings.paths.items()}
    elif args.command == "doctor":
        result = doctor(settings)
    elif args.command == "download-model":
        result = download_model(settings, weights=args.weights)
    else:
        from .inference import run_evaluation, run_infer
        result = (run_infer if args.command == "infer" else run_evaluation)(settings, args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
