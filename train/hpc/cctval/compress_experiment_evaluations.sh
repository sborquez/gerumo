#!/bin/bash
#SBATCH -p gpum
#SBATCH -J compress 
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_compress_%j.log
#SBATCH -e logs/error_compress_%j.log
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /data/atlas/dbetalhc/cta-test/gerumo/env/gerumo
echo "Running evaluate_assembler.sh"
echo ""

python_script="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/rzip.py"

# run assembler evaluation
function run_compression {
    folder=$1
    echo "folder: $folder"
    python $python_script -f "$folder" -v
}

echo "Compressing"
parent_folder="/data/atlas/dbetalhc/cta-test/gerumo/output/energy/baseline"
experiments=(
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Full_evaluation"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Loose_evaluation"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Tight_evaluation"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Full_evaluation_81"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Loose_evaluation_81"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Tight_evaluation_81"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Full_evaluation_243"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Loose_evaluation_243"
    # "UMONNA_ASSEMBLER_EVALUATIONS/UMONNA_Assembler_Tight_evaluation_243"
    # "UMONNA_UNIT_LST_Shallow_20210111_070459_5b2"
    # "UMONNA_UNIT_LST_Shallow_20210122_110300_5f3"
    # "UMONNA_UNIT_MST_Shallow_20210119_133656_5d4"
    # "UMONNA_UNIT_SST_Shallow_20210119_133602_b16"
    #
    # "UMONNA_UNIT_SST_FL_20210412_160513_9fb"
    # "UMONNA_UNIT_SST_CE_20210330_002344_193"
    # "UMONNA_UNIT_MST_FL_20210408_141047_fd5"
    # "UMONNA_UNIT_MST_CE_20210330_002339_9f8"
    # "UMONNA_UNIT_LST_FL_20210330_002339_b77"
    # "UMONNA_UNIT_LST_CE_20210330_002339_855"
    "CE_FL_ASSEMBLER_EVALUATIONS/CE_Assembler_TIght_evaluation"
    "CE_FL_ASSEMBLER_EVALUATIONS/CE_Assembler_Loose_evaluation"
    "CE_FL_ASSEMBLER_EVALUATIONS/CE_Assembler_Full_evaluation"
    "CE_FL_ASSEMBLER_EVALUATIONS/FL_Assembler_Tight_evaluation"
    "CE_FL_ASSEMBLER_EVALUATIONS/FL_Assembler_Loose_evaluation"
    "CE_FL_ASSEMBLER_EVALUATIONS/FL_Assembler_Full_evaluation"
)

for experiment_folder in "${experiments[@]}";do
    run_compression "$parent_folder/$experiment_folder"
done
