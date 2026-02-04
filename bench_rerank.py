import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests

from resources import SharedResources
from model_utils import rerank, score


@dataclass
class OneCall:
    ok: bool
    latency_ms: float
    err: Optional[str] = None


def percentile(sorted_vals: List[float], p: float) -> float:
    """
    Nearest-rank percentile.
    - sorted_vals must be sorted ascending.
    - p in [0, 100]
    """
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    # nearest-rank: rank = ceil(p/100 * N)
    n = len(sorted_vals)
    rank = int((p / 100.0) * n)
    if (p / 100.0) * n != rank:
        rank += 1
    idx = max(1, min(rank, n)) - 1
    return sorted_vals[idx]


def make_docs(n_docs: int) -> List[str]:
    base = [
        "Shanghai is a large city in China.",
        "The capital of China is Beijing.",
        "Guangzhou is a major city in southern China.",
        "China has a long history and rich culture.",
        "Beijing is known for the Forbidden City.",
    ]
    return [base[i % len(base)] + f" (doc_id={i})" for i in range(n_docs)]


def summarize(tag: str, calls: List[OneCall], wall_s: float) -> Dict[str, Any]:
    total = len(calls)
    ok_calls = [c for c in calls if c.ok]
    lat = sorted([c.latency_ms for c in ok_calls])

    ok_n = len(ok_calls)
    err_n = total - ok_n

    out: Dict[str, Any] = {
        "tag": tag,
        "total": total,
        "ok": ok_n,
        "err": err_n,
        "success_rate": (ok_n / total) if total else 0.0,
        "rps": (total / wall_s) if wall_s > 0 else 0.0,
        "avg_ms": statistics.mean(lat) if lat else float("nan"),
        "p50_ms": percentile(lat, 50) if lat else float("nan"),
        "p95_ms": percentile(lat, 95) if lat else float("nan"),
        "p99_ms": percentile(lat, 99) if lat else float("nan"),
        "max_ms": max(lat) if lat else float("nan"),
    }

    if err_n:
        sample = next(c for c in calls if not c.ok)
        out["err_sample"] = (sample.err or "")[:200]
    return out


def print_summary(s: Dict[str, Any]) -> None:
    print(
        f"[{s['tag']}] "
        f"ok={int(s['ok'])}/{int(s['total'])} "
        f"succ={s['success_rate']*100:.1f}% "
        f"rps={s['rps']:.2f} "
        f"avg={s['avg_ms']:.1f}ms "
        f"p50={s['p50_ms']:.1f}ms "
        f"p95={s['p95_ms']:.1f}ms "
        f"p99={s['p99_ms']:.1f}ms "
        f"max={s['max_ms']:.1f}ms"
    )
    if "err_sample" in s:
        print(f"  err_sample: {s['err_sample']}")


def new_client_like_shared() -> requests.Session:
    """
    requests.Session is not strictly thread-safe.
    For benchmarks, create one Session per thread for more stable results.

    We reuse base_url/timeout from SharedResources' singleton client (if present).
    """
    base = "http://127.0.0.1:11437/v1"
    timeout_s = 30.0

    shared = SharedResources.get_local_rerank_client()
    if hasattr(shared, "base_url"):
        base = str(shared.base_url)
    if hasattr(shared, "timeout_s"):
        timeout_s = float(shared.timeout_s)

    s = requests.Session()
    s.base_url = base.rstrip("/")
    s.timeout_s = timeout_s
    return s


def run_concurrency_perf_test(
    concurrency_levels: Tuple[int, ...] = (50, 100, 150),
    requests_per_user: int = 1,
    n_docs: int = 32,
    top_n: int = 10,
    timeout_s: float = 30.0,
    warmup: int = 2,
) -> None:
    """
    Concurrency benchmark for model_utils.rerank / model_utils.score.
    - concurrency = number of worker threads
    - total requests = concurrency * requests_per_user
    """
    query = "What is the capital of China?"
    documents = make_docs(n_docs)

    # Warm-up (serial) to reduce cold-start noise
    warm_client = SharedResources.get_local_rerank_client()
    for _ in range(warmup):
        rerank(warm_client, query=query, documents=documents, top_n=top_n, timeout_s=timeout_s)
        score(warm_client, model="qwen3-reranker", query=query, documents=documents, timeout_s=timeout_s)

    def worker(kind: str) -> List[OneCall]:
        client = new_client_like_shared()
        out: List[OneCall] = []
        for _ in range(requests_per_user):
            t0 = time.perf_counter()
            try:
                if kind == "rerank":
                    ranked, _, _ = rerank(
                        client, query=query, documents=documents, top_n=top_n, timeout_s=timeout_s
                    )
                    if not ranked:
                        raise RuntimeError("empty rerank result")
                else:
                    _, scores_aligned, _ = score(
                        client, model="qwen3-reranker", query=query, documents=documents, timeout_s=timeout_s
                    )
                    if len(scores_aligned) != len(documents):
                        raise RuntimeError(f"score len mismatch: {len(scores_aligned)} != {len(documents)}")

                dt = (time.perf_counter() - t0) * 1000.0
                out.append(OneCall(True, dt))
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                out.append(OneCall(False, dt, repr(e)))
        return out

    for conc in concurrency_levels:
        total_reqs = conc * requests_per_user

        # ---- rerank ----
        t0 = time.perf_counter()
        rerank_calls: List[OneCall] = []
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(worker, "rerank") for _ in range(conc)]
            for f in as_completed(futs):
                rerank_calls.extend(f.result())
        wall = time.perf_counter() - t0
        s1 = summarize(f"rerank | conc={conc} | total={total_reqs}", rerank_calls, wall)

        # ---- score ----
        t0 = time.perf_counter()
        score_calls: List[OneCall] = []
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(worker, "score") for _ in range(conc)]
            for f in as_completed(futs):
                score_calls.extend(f.result())
        wall = time.perf_counter() - t0
        s2 = summarize(f"score  | conc={conc} | total={total_reqs}", score_calls, wall)

        print_summary(s1)
        print_summary(s2)
        print("-" * 110)


if __name__ == "__main__":
    run_concurrency_perf_test(
        concurrency_levels=(50, 100, 150),
        requests_per_user=5,
        n_docs=16,   # try 16/32/64 to match your real workload
        top_n=10,
        timeout_s=30.0,
        warmup=2,
    )
