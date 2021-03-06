#!/bin/bash
#SBATCH  --output=/scratch_net/biwidl204/vramasamy/DAFormer_Panoptic/sbatch_log/run_panoptic_13.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=70G

source /itet-stor/vramasamy/net_scratch/conda/etc/profile.d/conda.sh
nvidia-smi
cd /scratch_net/biwidl204/vramasamy/DAFormer_Panoptic
conda activate daf
bash test.sh work_dirs/local-exp8/220611_2243_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_panoptic_sepaspp_mitb5_poly10warm_s0_ae75c
cd /scratch/
rmdir synthia/
rmdir cityscapes/
