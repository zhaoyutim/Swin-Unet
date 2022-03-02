#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  150GB
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 16

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate myenv
PYTHONUNBUFFERED=1; python3 train.py --dataset Palsar --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path . --max_epochs 150 --output_dir . --img_size 224 --base_lr 0.001 --batch_size 24