#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J bl_cd_energy
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

source activate /user/s/sborquez/envs/gerumo
echo "Running train_models.sh"
echo ""
cd /user/s/sborquez/gerumo/train

#TODO: add list of files
#configuration_files = 0 

#config="/user/s/sborquez/gerumo/train/config/cctval/energy/baseline/cd_sst.json"
#python train_model.py --config $config --quiet

#config="/user/s/sborquez/gerumo/train/config/cctval/energy/baseline/cd_mst.json"
#python train_model.py --config $config --quiet

config="/user/s/sborquez/gerumo/train/config/cctval/energy/baseline/cd_lst.json"
python train_model.py --config $config --quiet