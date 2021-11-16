#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J gerumo_debug
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_gerumo_debug_%j.log
#SBATCH -e logs/error_gerumo_debug_%j.log 
#SBATCH --gres=gpu:1

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /data/atlas/dbetalhc/cta-test/gerumo/env/gerumo
echo "Running test_gpu.sh"
echo ""

cd /data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train
python debug.py --gpu
