#!/bin/bash
#SBATCH -p gpum
#SBATCH -J assembler_bmo
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_assemblers_%j.log
#SBATCH -e logs/error_assemblers_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
echo "Running evaluate_assembler.sh"
echo ""
cd /user/s/sborquez/gerumo/train
#echo "Umonna Assemblers"
#folder="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/umonna_assembler_evaluations"
#python evaluate.py --assembler "$folder/umonna_assembler.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/umonna_assembler_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/umonna_assembler_lst_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/umonna_assembler_mst_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/umonna_assembler_sst_hillas.json" --all --samples --predictions --results

echo "BMO Assembler"
folder="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/bmo_assembler_evaluations"
python evaluate.py --assembler "$folder/bmo_assembler.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/bmo_assembler_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/bmo_assembler_lst_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/bmo_assembler_mst_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/bmo_assembler_sst_hillas.json" --all --samples --predictions --results


#echo "MVE Assembler"
#python evaluate.py --assembler ./config/cctval/alt_az/baseline/mve_assembler.json --all --samples --results 

#echo "CNN Deterministic Assembler"
#folder="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/cd_assembler_evaluations"
#python evaluate.py --assembler "$folder/cd_assembler.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/cd_assembler_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/cd_assembler_lst_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/cd_assembler_mst_hillas.json" --all --samples --predictions --results
#python evaluate.py --assembler "$folder/cd_assembler_sst_hillas.json" --all --samples --predictions --results


#echo "Testing assemblers"
#python evaluate.py --assembler ./config/cctval/alt_az/baseline/tester_assembler.json --samples --predictions --results
