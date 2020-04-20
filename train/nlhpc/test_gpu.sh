#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -J test
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=4000
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o test_%j.out
#SBATCH -e test_%j.err

#-----------------Toolchain---------------------------
ml purge
ml fosscuda/2019b
# ----------------MÃ³dulos-----------------------------
ml  CUDA/10.1.105 HDF5/1.10.4   Miniconda3/4.5.12   cuDNN/7.4.2.24  
# ----------------Comandos--------------------------

source activate /home/sborquez/envs/gerumo

#conda list

echo "Running test_gpu.sh"
echo ""

cd /home/sborquez/gerumo/train
python debug.py --gpu
