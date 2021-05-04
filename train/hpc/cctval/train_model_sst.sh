#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J bl_um_en_s
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_baseline_umo_sst_%j.log
#SBATCH -e logs/error_baseline_umo_sst_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=10-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------
source activate /data/atlas/dbetalhc/cta-test/gerumo/env/gerumo
cd /data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train

# User defined variables
## ALT AZ
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/alt_az/baseline/umonna_sst.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/alt_az/baseline/cd_sst.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/alt_az/baseline/bmo_sst.json"
## ENERGY
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/umonna_sst_mc_energy.json"
config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/umonna_sst.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/cd_sst.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/bmo_sst.json"

echo "config file: $config"
echo "Running train_model_sst.sh"
echo ""

# train 
python train_model.py --config $config --quiet
