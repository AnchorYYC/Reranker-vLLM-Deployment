import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

def _to_int(v: Optional[str], default: int) -> int:
    if v is None or v == "":
        return default
    return int(v)


def _to_float(v: Optional[str], default: float) -> float:
    if v is None or v == "":
        return default
    return float(v)


def _to_str(v: Optional[str], default: str) -> str:
    if v is None or v == "":
        return default
    return v


def _to_bool(v: Optional[str], default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class EnvConfig:
    # Model
    model_path_tiny: str
    model_path: str

    # vLLM runtime
    gpu_memory_utilization: float
    max_model_len: int

def load_env(dotenv_path: str = ".env", override: bool = True) -> EnvConfig:
    """
    Load config from .env (KEY=VALUE style).
    - override=True: .env Covering system environment variables
    """
    load_dotenv(dotenv_path=dotenv_path, override=override)

    model_path_tiny = _to_str(os.getenv("MODEL_PATH_TINY"), "/path/to/Qwen3-Reranker-0.6B/")
    model_path = _to_str(os.getenv("MODEL_PATH"), "/path/to/Qwen3-Reranker-4B or 8B/")

    gpu_memory_utilization = _to_float(os.getenv("GPU_MEMORY_UTILIZATION"), 0.90)
    max_model_len = _to_int(os.getenv("MAX_MODEL_LEN"), 4096)

    return EnvConfig(
        model_path_tiny=model_path_tiny,
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

CFG = load_env()
