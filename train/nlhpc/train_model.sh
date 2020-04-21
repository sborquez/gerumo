#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -J train
#SBATCH -p gpus
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=20
#SBATCH -c 2
#SBATCH --mem=6000
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err

#-----------------Toolchain---------------------------
ml purge
ml fosscuda/2019b
# ----------------MÃ³dulos-----------------------------
ml  CUDA/10.1.105 HDF5/1.10.4   Miniconda3/4.5.12   cuDNN/7.4.2.24  
# ----------------Comandos--------------------------

source activate /home/sborquez/envs/gerumo

#conda list

echo "Running train_model.sh"
echo ""

cd /home/sborquez/gerumo/train
python train_model.py
