import os

import pytest

from llm_jp_logkv.settings import load_settings


@pytest.fixture
def isolated_env(monkeypatch):
    # Restore even variables added directly by load_settings during each test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in (
        "DATA_DIR", "MODEL_DIR", "DATASET_DIR", "CHECKPOINT_DIR", "EXPERIMENT_DIR",
        "LOG_DIR", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME",
        "UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "LOGKV_SOURCE_DIR", "BASE_MODEL_ID",
        "BASE_MODEL_REVISION", "LLM_JP_LOGKV_ENV_FILE",
    ):
        os.environ.pop(name, None)


def write_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        f"DATA_DIR={tmp_path}/raid\n"
        "HF_HOME=${DATA_DIR}/cache/huggingface\n"
        "HF_HUB_CACHE=${HF_HOME}/hub\n"
        f"LOGKV_SOURCE_DIR={tmp_path}/reference\n"
        f"BASE_MODEL_REVISION={'a' * 40}\n"
    )
    return path


def test_dotenv_storage_is_explicit_and_independent_of_cwd(tmp_path, monkeypatch, isolated_env):
    env_file = write_env(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings = load_settings(env_file)
    assert settings.paths["HF_HUB_CACHE"] == tmp_path / "raid/cache/huggingface/hub"
    assert settings.model_path.is_relative_to(tmp_path / "raid/models")
    assert not settings.paths["DATA_DIR"].exists()
    settings.prepare_storage()
    assert all(path.is_dir() for path in settings.paths.values())
    assert not settings.source_dir.exists()  # A reference path is never created.


def test_project_dotenv_overrides_inherited_cache_and_interpolation(tmp_path, monkeypatch, isolated_env):
    env_file = write_env(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "override"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "global-hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "global-hf/hub"))
    settings = load_settings(env_file)
    assert settings.paths["DATA_DIR"] == tmp_path / "raid"
    assert settings.paths["HF_HOME"] == tmp_path / "raid/cache/huggingface"
    assert settings.paths["HF_HUB_CACHE"] == tmp_path / "raid/cache/huggingface/hub"


def test_missing_data_dir_never_falls_back_to_source_tree(tmp_path, isolated_env):
    with pytest.raises(ValueError, match="DATA_DIR"):
        load_settings(tmp_path / "missing.env")


@pytest.mark.parametrize("name,value", [("MODEL_DIR", "./models"), ("BASE_MODEL_REVISION", "main")])
def test_rejects_relative_storage_and_moving_revision(tmp_path, monkeypatch, isolated_env, name, value):
    env_file = write_env(tmp_path)
    with env_file.open("a") as stream:
        stream.write(f"{name}={value}\n")
    with pytest.raises(ValueError, match=name):
        load_settings(env_file)
