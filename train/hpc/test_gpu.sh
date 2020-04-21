#!/bin/bash
#PBS -q gpuk
#PBS -N test
#PBS -l walltime=168:00:00

# ----------------MÃ³dulos-----------------------------
cd $PBS_O_WORKDIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running test_gpu.sh"
echo ""

cd /home/sborquez/gerumo/train
python debug.py --gpu
