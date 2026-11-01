"""Load local configuration before importing torch or Hugging Face packages."""

from dataclasses import dataclass
import os
from pathlib import Path
import re

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    paths: dict[str, Path]
    source_dir: Path
    model_id: str
    model_revision: str

    @property
    def model_path(self) -> Path:
        return self.paths["MODEL_DIR"] / self.model_id.replace("/", "--") / self.model_revision

    def prepare_storage(self) -> None:
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)


def _absolute_path(name: str, default: Path | None = None) -> Path:
    value = os.environ.get(name, str(default) if default is not None else "")
    if not value:
        raise ValueError(f"Set {name} in .env or the environment")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {value}")
    return path


def load_settings(env_file: Path | None = None) -> Settings:
    """Project .env wins over inherited settings; no directories are created."""
    env_file = env_file or Path(os.environ.get("LLM_JP_LOGKV_ENV_FILE", ".env"))
    # Match scripts/uv.sh even when the login shell already defines HF_HOME.
    # Otherwise a plain `uv run --env-file .env` can silently use another cache.
    load_dotenv(env_file, override=True)
    root = _absolute_path("DATA_DIR")
    paths = {"DATA_DIR": root}
    for name, suffix in (
        ("MODEL_DIR", "models"),
        ("DATASET_DIR", "datasets"),
        ("CHECKPOINT_DIR", "checkpoints"),
        ("EXPERIMENT_DIR", "experiments"),
        ("LOG_DIR", "logs"),
        ("HF_HOME", "cache/huggingface"),
        ("TORCH_HOME", "cache/torch"),
        ("UV_CACHE_DIR", "cache/uv"),
        ("UV_PROJECT_ENVIRONMENT", "venvs/py312"),
    ):
        paths[name] = _absolute_path(name, root / suffix)
    paths["HF_HUB_CACHE"] = _absolute_path("HF_HUB_CACHE", paths["HF_HOME"] / "hub")
    paths["HF_DATASETS_CACHE"] = _absolute_path("HF_DATASETS_CACHE", paths["HF_HOME"] / "datasets")
    for name, path in paths.items():
        os.environ[name] = str(path)
    source = _absolute_path("LOGKV_SOURCE_DIR")
    model_id = os.environ.get("BASE_MODEL_ID", "llm-jp/llm-jp-4.1-8b-thinking")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
        raise ValueError("BASE_MODEL_ID must be a namespace/model Hub ID")
    revision = os.environ.get("BASE_MODEL_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("BASE_MODEL_REVISION must be a pinned 40-character commit SHA")
    return Settings(paths, source, model_id, revision)
