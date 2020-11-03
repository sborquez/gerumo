#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J assembler_eval 
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

# run assembler evaluation
function run_evaluation {
    folder=$1
    config_file=$2
    echo "configuration: $config_file"
    python evaluate.py --assembler "$folder/$config_file.json" --all --samples --predictions --results
}

# echo "ALT AZ"
# echo "======="
# echo "Umonna Assemblers"
# folder="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/umonna_assembler_evaluations"
# configurations=(
#     "umonna_assembler"
#     "umonna_assembler_hillas"
#     "umonna_assembler_lst_hillas"
#     "umonna_assembler_mst_hillas"
#     "umonna_assembler_sst_hillas"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

# echo "BMO Assembler"
# folder="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/bmo_assembler_evaluations"
# configurations=(
#     "bmo_assembler"
#     "bmo_assembler_hillas"
#     "bmo_assembler_lst_hillas"
#     "bmo_assembler_mst_hillas"
#     "bmo_assembler_sst_hillas"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

# echo "CNN Deterministic Assembler"
# folder="/user/s/sborquez/gerumo/train/config/cctval/alt_az/baseline/cd_assembler_evaluations"
# configurations=(
#     "cd_assembler"
#     "cd_assembler_hillas"
#     "cd_assembler_lst_hillas"
#     "cd_assembler_mst_hillas"
#     "cd_assembler_sst_hillas"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

echo "Energy"
echo "======"
# echo "Umonna Assemblers"
# folder="/user/s/sborquez/gerumo/train/config/cctval/energy/baseline/umonna_assembler_evaluations"
# configurations=(
#     "umonna_assembler"
#     "umonna_assembler_hillas"
#     "umonna_assembler_lst_hillas"
#     "umonna_assembler_mst_hillas"
#     "umonna_assembler_sst_hillas"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

# echo "BMO Assembler"
# folder="/user/s/sborquez/gerumo/train/config/cctval/energy/baseline/bmo_assembler_evaluations"
# configurations=(
#     "bmo_assembler"
#     "bmo_assembler_hillas"
#     "bmo_assembler_lst_hillas"
#     "bmo_assembler_mst_hillas"
#     "bmo_assembler_sst_hillas"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

echo "CNN Deterministic Assembler"
folder="/user/s/sborquez/gerumo/train/config/cctval/energy/baseline/cd_assembler_evaluations"
configurations=(
    #"cd_assembler"
    "cd_assembler_hillas"
    #"cd_assembler_lst_hillas"
    "cd_assembler_mst_hillas"
    "cd_assembler_sst_hillas"
)
for config_file in "${configurations[@]}";do
    run_evaluation $folder $config_file
done