"""Configuration: one YAML file + .env, loaded into pydantic models.

``${VAR}`` and ``${VAR:-default}`` placeholders in the YAML are interpolated from
the process environment (with .env loaded first) before validation.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_PLACEHOLDER = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _interpolate(value: object) -> object:
    """Recursively replace ${VAR} / ${VAR:-default} in strings using the env."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")
        return _PLACEHOLDER.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


class DatabaseConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


class DoclingConfig(BaseModel):
    mode: str = "remote_vlm"  # remote_vlm | local_vlm | standard
    vlm_api_base: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""
    picture_description_api_base: str = ""
    picture_description_api_key: str = ""
    picture_description_model: str = ""
    image_scale: float = 2.0
    batch_size: int = 4
    output_dir: str = "output"
    # standard-mode only: OCR is off by default (born-digital PDFs carry a text
    # layer; scanned inputs use the remote VLM instead). Formula enrichment adds a
    # model download, so it is opt-in.
    do_ocr: bool = False
    do_formula_enrichment: bool = False


class EmbeddingConfig(BaseModel):
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    dimensions: int = 1024
    similarity: str = "cosine"


class StageConfig(BaseModel):
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    batch_size: int = 16
    max_concurrent: int = 2
    num_retries: int = 5
    timeout_s: float = 300.0
    # Assembler-only: token budgets for the main (commit) window and the read-only
    # context peeked past its trailing edge.
    main_window_tokens: int = 1500
    context_window_tokens: int = 400


class Config(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    stages: dict[str, StageConfig] = Field(default_factory=dict)

    def stage(self, name: str) -> StageConfig:
        return self.stages.get(name, StageConfig())


def _default_config_path() -> Path:
    env_path = os.environ.get("MATH_TRAINER_CONFIG")
    if env_path:
        return Path(env_path)
    # repo-root/config/math_trainer.yaml relative to this file
    return Path(__file__).resolve().parents[3] / "config" / "math_trainer.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Load, interpolate, and validate the configuration."""
    load_dotenv(override=False)
    cfg_path = Path(path) if path else _default_config_path()
    raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    interpolated = _interpolate(raw or {})
    return Config.model_validate(interpolated)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
