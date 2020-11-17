#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J eval_cl 
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

source activate /data/atlas/dbetalhc/cta-test/gerumo/env/gerumo
echo "Running train_model_sst.sh"
echo ""

# umonna
#experiment="/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/UMONNA_UNIT_SST_Shallow_f2c057" 
experiment="/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/UMONNA_UNIT_MST_Shallow_5f2107"
#experiment="/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/UMONNA_UNIT_LST_Shallow_92e4f6"

# cd
#experiment="/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/CNN_DET_UNIT_LST_AZ_ALT_ADAM_MAE_LC_CD15_CD11_31af42" 

# bmo
#experiment="/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/BMO_UNIT_LST_690b25"

cd /data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train
python evaluate.py --results --samples --experiment "$experiment"

