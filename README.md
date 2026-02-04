# Reranker vLLM Deployment

This repository provides a **production-ready vLLM deployment and Python client utilities**
 applicable to  **Reranker** model series (case tested Qwen-Reranker series).

It covers:

- **Cold start inference** (Python-only, no server)
- **Hot start deployment** (long-running vLLM service)
- Stable HTTP APIs: **`/v1/rerank`** and **`/v1/score`**
- POSIX `sh` startup script for server environments
- Reusable Python client wrappers

This project focuses on **correctness, reproducibility, and real-world usage**.

------

## Supported Models

This repository is designed for the **vLLM-Reranker** series (including but not limited to):

- Qwen3-Reranker series(0.6B / 4B / 8B)

Model weights should be downloaded **in advance**.

The official Qwen model page of this instance:

- Hugging Face: https://huggingface.co/Qwen
- ModelScope: https://modelscope.cn/models/

> ⚠️ Model weights are **not** committed to this repository.

------

## Project Structure

```
.
├── bench_rerank.py        # Concurrency / performance benchmark
├── env_utils.py           # Environment config loader
├── log/                   # vLLM runtime logs
├── main.py                # Example application
├── model_utils.py         # rerank / score helpers
├── resources.py           # HTTP client (session reuse)
├── start_rerank.sh        # vLLM service startup script (POSIX sh)
├── vllm_usage.py          # Official-style example (4B / 8B)
└── vllm_usage_tiny.py     # Official-style example (0.6B)
```

------

## Cold Start (Python Only)

Cold start mode is useful for:

- Validating model correctness
- Debugging
- Quick experiments

### Run

```
export CUDA_VISIBLE_DEVICES=0
python vllm_usage_tiny.py   # for 0.6B
# or
python vllm_usage.py        # for 4B / 8B
```

> Make sure to update `MODEL_PATH_TINY` or `MODEL_PATH` in `env_utils.py`
> to point to your local model directory.

This mode **does not start an HTTP server**.

------

## Hot Start (vLLM Service Mode)

Hot start mode is recommended for **production or high-concurrency workloads**.

### Step 1: Configure `start_rerank.sh`

Key parameters to review:

- `MODEL_PATH`
- `PORT` (must be unique)
- `CUDA_VISIBLE_DEVICES`
- `TP` (must match the number of visible GPUs)

> ⚠️ Python clients must use the **same PORT** as defined here.

------

### Step 2: Start the Service

```
sh start_rerank.sh start
```

Check status:

```
sh start_rerank.sh status
```

Tail logs:

```
sh start_rerank.sh tail
```

------

## HTTP APIs

Once the service is running, vLLM exposes, encapsulated in `model_utils.py` :

### `/v1/rerank`

- Server-side sorting
- Returns ranked documents with relevance scores

### `/v1/score`

- Batch scoring
- Returns scores aligned with input order (no sorting)

------

## Python Client Usage

`resources.py` defines the shared HTTP client.

Make sure it matches the service configuration:

```
# resources.py
PORT = 11438  # MUST match start_rerank.sh
LOCAL_BASE_URL = f"http://127.0.0.1:{PORT}/v1"
```

Example:

```
# main.py
from resources import SharedResources
from model_utils import rerank, score

items, scores, raw = rerank(
    query="What is the capital of China?",
    documents=[...],
    top_n=5,
)
```

------

## Start vLLM Without the Script

You can also start vLLM directly:

```
export CUDA_VISIBLE_DEVICES=0

nohup vllm serve "${MODEL_PATH}" \
  --served-model-name "qwen3-reranker" \
  --task score \
  --port "11438" \
  --tensor-parallel-size "1" \
  --gpu-memory-utilization "0.85" \
  --max-model-len "4096" \
  --dtype "auto" \
  --enable-prefix-caching \
  --max-num-batched-tokens "24576" \
  --max-num-seqs "256" \
  --uvicorn-log-level info \
  --disable-uvicorn-access-log \
  --disable-log-stats \
  > vllm.log 2>&1 &
```

------

## Notes on GPU Configuration

- `TP` **must match** the number of GPUs in `CUDA_VISIBLE_DEVICES`
- Increasing GPU count does **not always improve throughput**
- Key tuning knobs:
  - `MAX_NUM_BATCHED_TOKENS`
  - `MAX_NUM_SEQS`
  - `GPU_MEMORY_UTILIZATION`

Benchmark before scaling.

------

## Summary

This repository provides:

- A **correct and stable** way to use Reranker with vLLM
- Both **cold start** and **hot start** workflows
- Clean separation between deployment and client code
- Practical guidance for real systems

If you plan to deploy **Reranker in production**, this setup is a solid foundation(Tested Qwen3 Reranker series models).