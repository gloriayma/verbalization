#!/bin/bash
# The 5 runs of run_grid.sbatch on one multi-GPU box without Slurm (e.g. a RunPod pod): run i on GPU i, until stopped.
#
#   FAMILY=s bash inversion/run_pod.sh start [extra train.py flags]   # launch (or resume) all 5 in the background
#   bash inversion/run_pod.sh status                                  # which are alive, latest step/val line, GPU util
#   bash inversion/run_pod.sh stop                                    # SIGTERM: each run checkpoints, then exits
#
# Always `stop` before stopping/terminating the pod: pod shutdown doesn't signal these processes, so everything since the
# last periodic checkpoint (--ckpt_every, 1000 here) would be lost. `start` again resumes every run from its checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOGS=runs/logs
PIDS=$LOGS/pids

alive() {
  [[ -f $PIDS ]] || return 0
  while read -r pid; do
    kill -0 "$pid" 2>/dev/null && echo "$pid"
  done < "$PIDS"
  return 0
}

start() {
  [[ -n ${FAMILY:-} ]] || { echo "set FAMILY=w|d|s, e.g. FAMILY=s bash $0 start"; exit 1; }
  local running; running=$(alive)
  [[ -z $running ]] || { echo "already running (pids $(echo $running)); run '$0 stop' first"; exit 1; }
  local F=$FAMILY
  local RUNS=(
    "--init ${F}1 --source ${F}1"
    "--init ${F}1 --source ${F}2"
    "--init random --source ${F}2"
    "--init ${F}1 --source mix${F}12"
    "--init random --source mix${F}12"
  )
  local n_gpus; n_gpus=$(nvidia-smi -L | wc -l)
  (( n_gpus >= ${#RUNS[@]} )) || { echo "need ${#RUNS[@]} GPUs, found $n_gpus"; exit 1; }
  mkdir -p "$LOGS"
  : > "$PIDS"
  for i in "${!RUNS[@]}"; do
    local log=$LOGS/${F}_run$i.out
    # Logs are appended to, so a resumed run continues its file. Flags given to `start` come last and win.
    CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 nohup $PY -m inversion.train ${RUNS[$i]} --ckpt_every 1000 "$@" \
      >> "$log" 2>&1 &
    echo $! >> "$PIDS"
    echo "GPU $i  pid $!  ${RUNS[$i]} $*  -> $log"
  done
}

status() {
  echo "alive: $(alive | tr '\n' ' ')"
  for log in "$LOGS"/*.out; do
    [[ -f $log ]] || continue
    echo "== $log"
    grep -E '^(resumed|run |got signal|saved )' "$log" | tail -2
    grep -E '^\[step [0-9]+\] ' "$log" | tail -1 | cut -c1-220
    grep -E '^\[val step [0-9]+\]\[[a-z0-9]+\] loss' "$log" | tail -2
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
}

stop() {
  local pids; pids=$(alive)
  [[ -n $pids ]] || { echo "nothing running"; return 0; }
  kill -TERM $pids
  echo "sent SIGTERM to $(echo $pids); waiting for each run to checkpoint..."
  while [[ -n $(alive) ]]; do sleep 5; done
  grep -h 'got signal' "$LOGS"/*.out | tail -n "$(echo $pids | wc -w)"
  echo "all stopped; '$0 start' (same FAMILY) resumes them"
}

cmd=${1:-}
shift || true
case $cmd in
  start) start "$@" ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: FAMILY=w|d|s $0 start [train.py flags] | $0 status | $0 stop"; exit 1 ;;
esac
