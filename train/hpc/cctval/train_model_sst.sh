#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J bl_bmo_sst
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_baseline_bmo_sst_%j.log
#SBATCH -e logs/error_baseline_bmo_sst_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=10-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------
source activate /user/s/sborquez/envs/gerumo
cd /user/s/sborquez/gerumo/train

# User defined variables
#config="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/umonna_sst.json"
# config="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/cd_sst.json"
config="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/bmo_sst.json"

echo "config file: $config"
echo "Running train_model_sst.sh"
echo ""

# train 
python train_model.py --config $config --quiet
