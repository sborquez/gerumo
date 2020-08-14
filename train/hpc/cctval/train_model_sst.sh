#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J gerumo_train_sst
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_gerumo_train_sst_%j.log
#SBATCH -e error_gerumo_train_sst_%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_sst.sh"
echo ""
 
cd /user/s/sborquez/gerumo/train
#python train_model.py --config ./config/cctval/energy/umonna_sst_hpc_onecell.json
#python train_model.py --config ./config/cctval/alt_az/umonna_sst_hpc_onecell.json
#python train_model.py --config ./config/cctval/alt_az_energy/umonna_sst_hpc_onecell.json
#python train_model.py --config ./config/cctval/alt_az/umonna_sst_hpc_all.json
python train_model.py --config ./config/cctval/alt_az/inf578/umonna_sst.json --quiet -G
#python train_model.py --config ./config/cctval/energy/umonna_sst_hpc_all.json
