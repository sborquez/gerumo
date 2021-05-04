#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J sst_ce
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_baseline_det_energy_%j.log
#SBATCH -e logs/error_baseline_det_energy_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /data/atlas/dbetalhc/cta-test/gerumo/env/gerumo
echo "Running train_models.sh"
echo ""
cd /data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train

#TODO: add list of files
#configuration_files = 0 

#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/umonna_sst.json"
#python train_model.py --config $config --quiet

#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/umonna_mst.json"
#python train_model.py --config $config --quiet

#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/baseline/umonna_lst.json"

# Focal Loss
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/focal_loss/umonna_lst_fl.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/focal_loss/umonna_mst_fl.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/focal_loss/umonna_sst_fl.json"
# Cross Entropy
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/focal_loss/umonna_lst_ce.json"
#config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/focal_loss/umonna_mst_ce.json"
config="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train/config/cctval/energy/focal_loss/umonna_sst_ce.json"


python train_model.py --config $config --quiet