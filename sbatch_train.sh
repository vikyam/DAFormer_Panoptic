#!/bin/bash
#SBATCH  --output=/scratch_net/biwidl204/vramasamy/DAFormer_Panoptic/sbatch_log/run_panoptic_8.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G

source /itet-stor/vramasamy/net_scratch/conda/etc/profile.d/conda.sh
nvidia-smi
cd /scratch_net/biwidl204/vramasamy/DAFormer_Panoptic
conda activate daf
python run_experiments.py --exp=8
cd /scratch/
rmdir synthia/
rmdir cityscapes/
