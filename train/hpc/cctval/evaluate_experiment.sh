#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J tiny_eval 
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_tiny_eval_%j.log
#SBATCH -e logs/error_tiny_eval_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_sst.sh"
echo ""

experiment="/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/TINY_UNIT_MST_MAE_SIMPLE_d525a5" 
cd /user/s/sborquez/gerumo/train
python evaluate.py --results --samples --predictions --experiment "$experiment"

