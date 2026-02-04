from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import threading

import requests

# Consistent with the parameters in your startup script start_rerank.sh
PORT=11438

LOCAL_BASE_URL = f"http://127.0.0.1:{PORT}/v1"

@dataclass(frozen=True)
class RerankHttpClientConfig:
    base_url: str = LOCAL_BASE_URL
    timeout_s: float = 30.0


class SharedResources:
    """
    A lightweight singleton holder for HTTP clients (requests.Session).
    Reusing a Session improves performance via connection pooling.
    """

    _rerank: Optional[Tuple[requests.Session, RerankHttpClientConfig]] = None
    _lock = threading.Lock()

    @classmethod
    def get_local_rerank_client(
        cls,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> Tuple[requests.Session, RerankHttpClientConfig]:
        """
        Returns:
            (session, config)

        Notes:
        - base_url example: "http://127.0.0.1:11437/v1"
        - timeout_s is used as default per-request timeout
        """
        cfg = RerankHttpClientConfig(
            base_url=(base_url or LOCAL_BASE_URL).rstrip("/"),
            timeout_s=(30.0 if timeout_s is None else float(timeout_s)),
        )

        # Fast path: return existing client if config matches
        if cls._rerank is not None:
            s, old_cfg = cls._rerank
            if old_cfg == cfg:
                return s, old_cfg

        with cls._lock:
            if cls._rerank is not None:
                s, old_cfg = cls._rerank
                if old_cfg == cfg:
                    return s, old_cfg
                # Config changed: close old session
                try:
                    s.close()
                except Exception:
                    pass
                cls._rerank = None

            s = requests.Session()
            cls._rerank = (s, cfg)
            return s, cfg

    @classmethod
    def close_local_rerank_client(cls) -> None:
        with cls._lock:
            if cls._rerank is not None:
                s, _ = cls._rerank
                try:
                    s.close()
                except Exception:
                    pass
                cls._rerank = None
