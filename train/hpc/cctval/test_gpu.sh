#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J gerumo_debug
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_gerumo_debug_%j.log
#SBATCH -e error_gerumo_debug_%j.log 
#SBATCH --gres=gpu:1

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running test_gpu.sh"
echo ""

cd /user/s/sborquez/gerumo/train
python debug.py --gpu
