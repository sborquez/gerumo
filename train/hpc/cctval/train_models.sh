#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J bl_det_assembler
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_baseline_det_assembler_%j.log
#SBATCH -e logs/error_baseline_det_assembler_%j.log
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
configuration_files = 0 


python train_model.py --config ./config/cctval/alt_az/tiny_tests_2/tiny_lst_mae_simple.json --quiet
python train_model.py --config ./config/cctval/alt_az/tiny_tests_2/tiny_mst_mae_simple.json --quiet
python train_model.py --config ./config/cctval/alt_az/tiny_tests_2/tiny_sst_mae_simple.json --quiet