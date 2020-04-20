#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpus
#SBATCH -n 1
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=4365
#SBATCH --gres=gpu:1

pwd

ml CUDA/10.1.105
ml cuDNN/7.4.2.24
ml Miniconda3

source activate /home/sborquez/.envs/gerumo

#conda list

echo "Running test_gpu.sh"
echo ""

python ../debug.py --gpu