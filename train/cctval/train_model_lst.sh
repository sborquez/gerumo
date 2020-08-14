#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J gerumo_train_lst
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_gerumo_train_lst_%j.log
#SBATCH -e error_gerumo_train_lst_%j.log
#SBATCH --gres=gpu:1


# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_lst.sh"
echo ""
 
cd /user/s/sborquez/gerumo/train

#python train_model.py --config ./config/cctval/energy/umonna_lst_hpc_onecell.json
#python train_model.py --config ./config/cctval/energy/umonna_lst_hpc_all.json
#python train_model.py --config ./config/cctval/alt_az/umonna_lst_hpc_onecell.json
#python train_model.py --config ./config/cctval/alt_az_energy/umonna_lst_hpc_onecell.json
#python train_model.py --config ./config/cctval/alt_az/umonna_lst_hpc_all.json
python train_model.py --config ./config/cctval/alt_az/umonna_lst_hpc_hd_all.json
