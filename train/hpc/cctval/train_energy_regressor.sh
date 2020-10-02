#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_train
#SBATCH --mail-user=patricio.campana@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_preprocessing_%j.log
#SBATCH -e error_preprocessing_%j.log

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/c/campana/envs/gerumo
echo "Running train_energy_regressor.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/

RESULTS=${ML1PATH}baseline/
DXY=${RESULTS}/D$1$2/
mkdir -p ${DXY}

for i in {00$1..$2} do
    DATASET=${i}/
    d=${ML1PATH}${DATASET}
    RESULTS=${d}baseline/
    TRAIN_RESULTS=${RESULTS}train/

    echo "Training energy regressor ${DXY} for dataset ${d}"
    python train_energy_regressor.py -e ${d}events.csv -t ${d}telescopes.csv -H ${TRAIN_RESULTS}hillas.csv -r ${TRAIN_RESULTS}results_hillas.csv -o ${DXY}energy_regressor.pickle
done
