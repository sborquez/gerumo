#!/bin/bash
#SBATCH -p gpuk
#SBATCH -J eval
#SBATCH --mail-user=sebastian.borquez@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_cd_eval_%j.log
#SBATCH -e logs/error_cd_eval_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=7-0

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /data/atlas/dbetalhc/cta-test/gerumo/env/gerumo
echo "Running evaluate_assembler.sh"
echo ""

train_folder="/data/atlas/dbetalhc/cta-test/gerumo/src/gerumo/train"
cd $train_folder

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
# folder="$train_folder/config/cctval/alt_az/baseline/umonna_assembler_evaluations"
# configurations=(
#     "umonna_assembler_full"
#     "umonna_assembler_tight_all"
#     "umonna_assembler_loose_all"
#     "umonna_assembler_tight_lst"
#     "umonna_assembler_tight_mst"
#     "umonna_assembler_tight_sst"
#     "umonna_assembler_loose_lst"
#     "umonna_assembler_loose_sst"
#     "umonna_assembler_loose_mst"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

#echo "BMO Assembler"
#folder="$train_folder/config/cctval/alt_az/baseline/bmo_assembler_evaluations"
#folder="$train_folder/config/cctval/alt_az/BMO_assembler_methods"
#folder="$train_folder/config/cctval/alt_az/BMO_dropout_tests/bmo_assembler_evaluations"
#configurations=(
      #"bmo_assembler_full"
      #"bmo_assembler_tight_all"
      #"bmo_assembler_loose_all"
#     "bmo_assembler_tight_lst"
#     "bmo_assembler_tight_mst"
#     "bmo_assembler_tight_sst"
#     "bmo_assembler_loose_lst"
#     "bmo_assembler_loose_sst"
#     "bmo_assembler_loose_mst"
      #"bmo_assembler_kde"
      #"bmo_assembler_mean_var"
      #"bmo_assembler_norm_prod"
      #"bmo_assembler_wasserstein"
      #"bmo_assembler_tight_mst_d02"
      #"bmo_assembler_tight_mst_d05"
      #"bmo_assembler_tight_mst_d08"
      #"bmo_assembler_tight_mst_d09"
#)
#for config_file in "${configurations[@]}";do
#    run_evaluation $folder $config_file
#done

# echo "CNN Deterministic Assembler"
# folder="$train_folder/config/cctval/alt_az/baseline/cd_assembler_evaluations"
# configurations=(
    #"cd_assembler_full"
    # "cd_assembler_tight_all"
    # "cd_assembler_loose_all"
    # "cd_assembler_tight_lst"
    # "cd_assembler_tight_mst"
    # "cd_assembler_tight_sst"
    # "cd_assembler_loose_lst"
    # "cd_assembler_loose_sst"
    # "cd_assembler_loose_mst"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

# echo "Energy"
# echo "======" 
# echo "Umonna Assemblers"
# folder="$train_folder/config/cctval/energy/baseline/umonna_assembler_evaluations"
# configurations=(
#     # "umonna_assembler_full"
#     # "umonna_assembler_loose_all"
#     # "umonna_assembler_tight_all"
#     #"umonna_assembler_tight_no_lst"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

# echo "BMO Assembler"
# folder="$train_folder/config/cctval/energy/baseline/bmo_assembler_evaluations"
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
# folder="$train_folder/config/cctval/energy/baseline/cd_assembler_evaluations"
# configurations=(
#     "cd_assembler_full"
#     # "cd_assembler_loose_all"
#     # "cd_assembler_tight_all"
#     # "cd_assembler_lst_hillas"
#     # "cd_assembler_mst_hillas"
#     # "cd_assembler_sst_hillas"
# )
# for config_file in "${configurations[@]}";do
#     run_evaluation $folder $config_file
# done

echo "Focal Loss vs CrossEntropy"
echo "======" 
echo "FOCAL LOSS Assemblers"
folder="$train_folder/config/cctval/energy/focal_loss/FL_PRIOR_evaluations"
configurations=(
    "umonna_assembler_full"
    "umonna_assembler_loose_all"
    "umonna_assembler_tight_all"
)
for config_file in "${configurations[@]}";do
    run_evaluation $folder $config_file
done

echo "Cross Entropy Assemblers"
folder="$train_folder/config/cctval/energy/focal_loss/CE_PRIOR_evaluations"
configurations=(
    "umonna_assembler_full"
    "umonna_assembler_loose_all"
    "umonna_assembler_tight_all"
)
for config_file in "${configurations[@]}";do
    run_evaluation $folder $config_file
done
