#!/bin/sh
set -eu

cd "$(dirname "$0")" || exit 1

# ====== 配置（支持环境变量覆盖） ======
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

: "${LOG_DIR:=./log}"
: "${PORT:=11438}"
: "${MODEL_PATH:=/path/to/Qwen3-Reranker}"
: "${SERVED_MODEL_NAME:=qwen3-reranker}"

: "${DTYPE:=auto}"
: "${MAX_NUM_BATCHED_TOKENS:=24576}"
: "${MAX_NUM_SEQS:=256}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=4096}"
: "${TP:=1}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rerank_${SERVED_MODEL_NAME}_${PORT}.log"
PID_FILE="${LOG_DIR}/rerank_${SERVED_MODEL_NAME}_${PORT}.pid"

# 尽力提高 fd soft limit（失败不退出）
ulimit -n 1048576 2>/dev/null || true

# ====== 工具函数 ======
port_pid() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | awk -v p=":${PORT}" '
      $4 ~ p { match($0, /pid=([0-9]+)/, a); if (a[1]!="") { print a[1]; exit } }' || true
  else
    echo ""
  fi
}

is_alive() { kill -0 "$1" 2>/dev/null; }

running_pid() {
  # 优先 PID_FILE，其次端口兜底
  if [ -f "${PID_FILE}" ]; then
    P="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [ -n "${P}" ] && is_alive "${P}"; then
      echo "${P}"; return
    fi
  fi
  port_pid
}

show_cfg() {
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "PORT=${PORT} TP=${TP}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "DTYPE=${DTYPE} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "LOG_FILE=${LOG_FILE}"
}

do_start() {
  P="$(running_pid)"
  if [ -n "${P}" ]; then
    echo "[OK] VLLM ${PORT} 已在运行 (PID: ${P})"
    echo "[OK] 日志: ${LOG_FILE}"
    exit 0
  fi

  echo "启动 vLLM Rerank 服务中..."
  show_cfg

  nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --task score \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    --enable-prefix-caching \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --uvicorn-log-level info \
    --disable-uvicorn-access-log \
    --disable-log-stats \
    > "${LOG_FILE}" 2>&1 &
    
    # --enable-log-requests \
    # --max-log-len 2048 \

  NEW_PID=$!
  echo "${NEW_PID}" > "${PID_FILE}"
  sleep 1

  if is_alive "${NEW_PID}"; then
    echo "[OK] VLLM ${PORT} 已后台启动 (PID: ${NEW_PID})"
    echo "[OK] 日志: ${LOG_FILE}"
  else
    echo "[ERR] 启动失败（进程未存活）！请检查日志: ${LOG_FILE}"
    exit 1
  fi
}

do_stop() {
  P="$(running_pid)"
  if [ -z "${P}" ]; then
    echo "[OK] VLLM ${PORT} 未运行"
    exit 0
  fi

  echo "停止 VLLM ${PORT} (PID: ${P}) ..."

  # 1) 尝试杀进程组（能带走子进程/worker）
  # 获取 PGID：ps -o pgid= -p <pid>
  PG="$(ps -o pgid= -p "${P}" 2>/dev/null | tr -d ' ' || true)"

  if [ -n "${PG}" ]; then
    # TERM 整个进程组
    kill -TERM "-${PG}" 2>/dev/null || true
  else
    # 退化：TERM 单进程
    kill -TERM "${P}" 2>/dev/null || true
  fi

  # 2) 等待最多 5 秒，仍存活则 KILL
  i=0
  while [ $i -lt 5 ]; do
    if ! kill -0 "${P}" 2>/dev/null; then
      break
    fi
    sleep 1
    i=$((i + 1))
  done

  if kill -0 "${P}" 2>/dev/null; then
    echo "[WARN] TERM 后仍未退出，执行强制 kill -9 ..."
    if [ -n "${PG}" ]; then
      kill -KILL "-${PG}" 2>/dev/null || true
    else
      kill -KILL "${P}" 2>/dev/null || true
    fi
  fi

  # 3) 兜底：再杀一次“端口监听 PID”（有时 P 不是监听者）
  RUNP="$(port_pid)"
  if [ -n "${RUNP}" ]; then
    kill -KILL "${RUNP}" 2>/dev/null || true
  fi

  rm -f "${PID_FILE}" 2>/dev/null || true
  echo "[OK] stop done. 如仍有 GPU 进程占用，请 tail 日志确认是否还有残留 worker。"
}


do_status() {
  P="$(running_pid)"
  if [ -n "${P}" ]; then
    echo "[OK] VLLM ${PORT} 运行中 (PID: ${P})"
    echo "[OK] 日志: ${LOG_FILE}"
  else
    echo "[OK] VLLM ${PORT} 未运行"
  fi
}

do_tail() {
  touch "${LOG_FILE}"
  tail -n 200 -f "${LOG_FILE}"
}

# ====== 命令入口 ======
CMD="${1:-start}"
case "${CMD}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 1; do_start ;;
  status)  do_status ;;
  tail)    do_tail ;;
  cfg)     show_cfg ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|tail|cfg}"
    exit 2
    ;;
esac
