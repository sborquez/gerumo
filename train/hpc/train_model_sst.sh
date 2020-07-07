#!/bin/bash
#PBS -q gpuk
#PBS -N train_sst_all_e
#PBS -l walltime=168:00:00

# ----------------MÃ³dulos-----------------------------
cd $PBS_O_WORKDIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_sst.sh"
echo ""
 
cd /user/s/sborquez/gerumo/train
#python train_model.py --config ./config/hpc/energy/umonna_sst_hpc_onecell.json
#python train_model.py --config ./config/hpc/alt_az/umonna_sst_hpc_onecell.json
#python train_model.py --config ./config/hpc/alt_az_energy/umonna_sst_hpc_onecell.json
#python train_model.py --config ./config/hpc/alt_az/umonna_sst_hpc_all.json
python train_model.py --config ./config/hpc/alt_az/umonna_sst_hpc_hd_all.json
#python train_model.py --config ./config/hpc/energy/umonna_sst_hpc_all.json
