#!/bin/bash
# The 5 inversion experiments for one seed family, one GPU each, training until you scancel them.
#
#   cd ~/dev/verbalization && mkdir -p runs/slurm
#   sbatch inversion/run_grid.sbatch                              # weight seeds (w), layer 6
#   FAMILY=d sbatch inversion/run_grid.sbatch                     # data seeds: same init, different data order
#   FAMILY=s sbatch inversion/run_grid.sbatch --layer final       # combined seeds, extra train.py flags after the script
#   sbatch --array=3 inversion/run_grid.sbatch                    # just one of the 5
#
# Slurm's QOS caps jobs at 24h: 5 minutes before that, the job checkpoints and requeues itself, and the requeued job
# resumes from the checkpoint. `scancel` instead checkpoints and stops for good.
#SBATCH --job-name=inversion
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm/%x_%A_%a.out

F=${FAMILY:-w}
RUNS=(
  "--init ${F}1 --source ${F}1"
  "--init ${F}1 --source ${F}2"
  "--init random --source ${F}2"
  "--init ${F}1 --source mix${F}12"
  "--init random --source mix${F}12"
)

cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$SLURM_SUBMIT_DIR/.venv/bin/python}
export PYTHONUNBUFFERED=1
: "${INVERSION_DATA:?export INVERSION_DATA (the data directory) before sbatch}"

$PY -m inversion.train ${RUNS[$SLURM_ARRAY_TASK_ID]} "$@" &
PID=$!
trap 'echo "time limit near: checkpoint + requeue"; kill -TERM $PID; wait $PID;
      scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}' USR1
wait $PID
