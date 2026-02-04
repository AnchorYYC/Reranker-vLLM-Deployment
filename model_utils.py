from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from resources import SharedResources  # 如果你的 SharedResources 在 resources.py，同文件就删掉这行并直接引用

@dataclass(frozen=True)
class RerankItem:
    """One document's score aligned to the original documents list."""
    index: int   # index in input documents
    score: float
    doc: str     # document text


def _post_json(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    """POST JSON and return (json, client_cost_ms). Raises with readable error."""
    t0 = time.time()
    resp = session.post(url, json=payload, timeout=timeout_s)
    cost_ms = (time.time() - t0) * 1000.0

    if resp.status_code >= 400:
        # keep server error text; it is very helpful for debugging schema issues
        raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text}")

    try:
        raw = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from {url}: {e}; text={resp.text[:500]}") from e

    raw["_client_cost_ms"] = cost_ms
    return raw, cost_ms


def rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Tuple[List[RerankItem], List[Optional[float]], Dict[str, Any]]:
    """
    Call vLLM OpenAI-compatible rerank endpoint: POST /v1/rerank

    Args:
        query: query string
        documents: candidate documents
        top_n: return top N results (sorted by server)
        model: served model name (default from env_utils/SharedResources)
        base_url: e.g. "http://127.0.0.1:11436/v1"
        timeout_s: per-request timeout

    Returns:
        ranked_items: sorted by server (descending relevance)
        scores_aligned: aligned to input docs; not selected => None
        raw: original JSON (plus _client_cost_ms)
    """
    if not documents:
        return [], [], {"results": [], "usage": {"total_tokens": 0}}

    session, cfg = SharedResources.get_local_rerank_client(base_url=base_url, timeout_s=timeout_s)

    url = f"{cfg.base_url}/rerank"
    payload: Dict[str, Any] = {
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = int(top_n)

    raw, _ = _post_json(session, url, payload, timeout_s=(timeout_s or cfg.timeout_s))

    results = raw.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Unexpected /v1/rerank response (results not list): {raw}")

    ranked_items: List[RerankItem] = []
    scores_aligned: List[Optional[float]] = [None] * len(documents)

    for r in results:
        idx = int(r["index"])
        score = float(r.get("relevance_score", 0.0))
        doc_text = (r.get("document") or {}).get("text") or documents[idx]
        ranked_items.append(RerankItem(index=idx, score=score, doc=doc_text))
        if 0 <= idx < len(scores_aligned):
            scores_aligned[idx] = score

    return ranked_items, scores_aligned, raw


def score(
    query: str,
    documents: List[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Tuple[List[RerankItem], List[float], Dict[str, Any]]:
    """
    Call vLLM OpenAI-compatible score endpoint: POST /v1/score

    Verified schema:
      request payload:
        {"model": "...", "text_1": <str|list[str]>, "text_2": <list[str]>}

      typical response:
        {"data":[{"index":0,"score":...}, ...], "usage": {...}}

    Returns:
        items: in original order (NOT sorted)
        scores_aligned: list[float] aligned to documents
        raw: original JSON (plus _client_cost_ms)
    """
    if not documents:
        return [], [], {"data": [], "usage": {"total_tokens": 0}}

    session, cfg = SharedResources.get_local_rerank_client(base_url=base_url, timeout_s=timeout_s)

    url = f"{cfg.base_url}/score"
    payload: Dict[str, Any] = {
        "text_1": query,
        "text_2": documents,
    }

    raw, _ = _post_json(session, url, payload, timeout_s=(timeout_s or cfg.timeout_s))

    data = raw.get("data", [])
    if not isinstance(data, list):
        raise ValueError(f"Unexpected /v1/score response (data not list): {raw}")

    scores_aligned: List[Optional[float]] = [None] * len(documents)
    for x in data:
        idx = int(x["index"])
        sc = float(x["score"])
        if 0 <= idx < len(scores_aligned):
            scores_aligned[idx] = sc

    # Ensure full alignment
    if any(v is None for v in scores_aligned):
        raise ValueError(f"Incomplete scores returned by /v1/score: {raw}")

    scores_final: List[float] = [float(v) for v in scores_aligned]  # type: ignore[arg-type]
    items = [RerankItem(index=i, score=scores_final[i], doc=documents[i]) for i in range(len(documents))]
    return items, scores_final, raw
