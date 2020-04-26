#!/bin/bash
#PBS -q gpuk
#PBS -N train_mst
#PBS -l walltime=168:00:00

# ----------------MÃ³dulos-----------------------------
cd $PBS_O_WORKDIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_mst.sh"
echo ""
 
cd /user/s/sborquez/gerumo/train
#python train_model.py --config ./config/umonna_mst_hpc.json
python train_model.py --config ./config/umonna_mst_hpc_3d.json
#python train_model.py --config ./config/umonna_mst_hpc_onecell.json

