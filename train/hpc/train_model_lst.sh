#!/bin/bash
#PBS -q gpuk
#PBS -N train_lst_all_hd
#PBS -l walltime=168:00:00

# ----------------MÃ³dulos-----------------------------
cd $PBS_O_WORKDIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_lst.sh"
echo ""
 
cd /user/s/sborquez/gerumo/train

#python train_model.py --config ./config/hpc/energy/umonna_lst_hpc_onecell.json
#python train_model.py --config ./config/hpc/energy/umonna_lst_hpc_all.json
#python train_model.py --config ./config/hpc/alt_az/umonna_lst_hpc_onecell.json
#python train_model.py --config ./config/hpc/alt_az_energy/umonna_lst_hpc_onecell.json
#python train_model.py --config ./config/hpc/alt_az/umonna_lst_hpc_all.json
python train_model.py --config ./config/hpc/alt_az/umonna_lst_hpc_hd_all.json
