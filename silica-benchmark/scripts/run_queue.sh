#!/usr/bin/env bash
# Step-1 learning-curve queue: N = 10..300, seeds 0 -> 2, three approaches, one run at a time on the GPU.
# Resumable: a run is skipped when its results/<approach>/s<seed>_n<N>.json exists.
# Launch detached from the repo root so it survives the session and laptop sleep:
#   setsid nohup bash silica-benchmark/scripts/run_queue.sh &>> silica-benchmark/runs/queue.log &
# Progress: ls silica-benchmark/results/*/   or   tail silica-benchmark/runs/queue.log
set -u
cd "$(dirname "$0")/../.."
B=silica-benchmark
MACE=$B/.venv-mace/bin/python
SCHNET=$B/.venv-schnet/bin/python
mkdir -p $B/runs

run() {  # run <result-json> <log> <cmd...>
    local out=$1 log=$2; shift 2
    if [[ -f $out ]]; then echo "$(date '+%F %T') skip $out"; return; fi
    echo "$(date '+%F %T') start $out"
    if "$@" &> "$log"; then echo "$(date '+%F %T') done  $out"; else echo "$(date '+%F %T') FAILED $out (see $log)"; fi
}

for seed in ${SEEDS:-0 1 2}; do  # SEEDS="1" runs a subset
    for n in 10 30 100 300; do
        run $B/results/schnet/s${seed}_n${n}.json            $B/runs/schnet_s${seed}_n${n}.log  $SCHNET $B/scripts/train_schnet.py $seed $n
        run $B/results/mace_mp0_finetune/s${seed}_n${n}.json $B/runs/ft_s${seed}_n${n}.log      $MACE $B/scripts/train_mace.py ft $seed $n
        run $B/results/mace_scratch/s${seed}_n${n}.json      $B/runs/scratch_s${seed}_n${n}.log $MACE $B/scripts/train_mace.py scratch $seed $n
    done
done
echo "$(date '+%F %T') queue finished"
