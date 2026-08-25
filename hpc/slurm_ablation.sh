#!/bin/bash
# One array task per cell of the ablation grid.
#
# The grid is embarrassingly parallel across cells and each cell is single-node,
# so an array job is the right shape and a distributed framework is not. Cohort
# generation is done once, before the array, because 140 tasks racing to write
# the same slides is the classic way to spend a night debugging a filesystem.
#
#SBATCH --job-name=wsi-ablation
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load Python/3.12 || true
source "${VENV_PATH:-.venv}/bin/activate"

TISSUE=(threshold threshold threshold threshold threshold threshold unetpp unetpp unetpp unetpp unetpp unetpp)
COLOUR=(none none macenko macenko physical physical none none macenko macenko physical physical)
ENCODER=(task-trained fixed-bank task-trained fixed-bank task-trained fixed-bank task-trained fixed-bank task-trained fixed-bank task-trained fixed-bank)

I=${SLURM_ARRAY_TASK_ID}
CELL="${TISSUE[$I]}-${COLOUR[$I]}-${ENCODER[$I]}"
CONFIG="configs/generated/${CELL}.yaml"
mkdir -p configs/generated logs

# Threads are pinned because torch will otherwise take every core on the node
# and twelve tasks doing that at once run slower than one.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python - "$CONFIG" "${TISSUE[$I]}" "${COLOUR[$I]}" "${ENCODER[$I]}" <<'PY'
import sys, yaml, pathlib
path, tissue, colour, encoder = sys.argv[1:5]
config = yaml.safe_load(pathlib.Path("configs/default.yaml").read_text())
config["run"]["tissue_arms"] = [tissue]
config["run"]["colour_arms"] = [colour]
config["run"]["encoder_arms"] = [encoder]
config["run"]["out_dir"] = f"runs/{tissue}-{colour}-{encoder}"
pathlib.Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
PY

python -m wsi_ablation.cli --config "$CONFIG" run
