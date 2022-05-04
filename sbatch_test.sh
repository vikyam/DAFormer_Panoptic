#!/bin/bash
#SBATCH  --output=/scratch_net/biwidl204/vramasamy/DAFormer/sbatch_log/run_1.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=12G

source /itet-stor/vramasamy/net_scratch/conda/etc/profile.d/conda.sh
nvidia-smi
cd /scratch_net/biwidl204/vramasamy/DAFormer
conda activate daf
python -u run_experiments.py --exp 7
