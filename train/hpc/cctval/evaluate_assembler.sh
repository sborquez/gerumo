#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J assembler_umonna 
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_assemblers_%j.log
#SBATCH -e logs/error_assemblers_%j.log
#SBATCH --gres=gpu:2
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running train_model_mst.sh"
echo ""
 
cd /user/s/sborquez/gerumo/train
#echo "Umonna Assembler"
#python evaluate.py --assembler ./config/cctval/alt_az/baseline/umonna_assembler.json --all --samples --predictions --results

#echo "BMO Assembler"
#python evaluate.py --assembler ./config/cctval/alt_az/baseline/bmo_assembler.json --all --samples --results 

#echo "MVE Assembler"
#python evaluate.py --assembler ./config/cctval/alt_az/baseline/mve_assembler.json --all --samples --predictions --results 

echo "CNN Deterministic Assembler"
python evaluate.py --assembler ./config/cctval/alt_az/baseline/cd_assembler.json --all --samples --predictions --results
