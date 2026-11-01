"""Storage and dependency preflight; model conversion is a separate next stage."""

import argparse
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
    (settings.model_path / "download_manifest.json").write_text(
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
    args = parser.parse_args()
    try:
        settings = load_settings(args.env_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "prepare-storage":
        settings.prepare_storage()
        result = {name: str(path) for name, path in settings.paths.items()}
    elif args.command == "doctor":
        result = doctor(settings)
    else:
        result = download_model(settings, weights=args.weights)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
