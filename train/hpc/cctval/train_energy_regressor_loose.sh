#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_train
#SBATCH --mail-user=patricio.campana@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_energy_training_loose_%j.log
#SBATCH -e error_energy_training_loose_%j.log
#SBATCH --time 48:00:00

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/c/campana/envs/gerumo
echo "Running train_energy_regressor.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/
HILLAS=/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/HILLAS/
TRAIN_DATA=${HILLAS}energy/train/loose/

for i in $(seq -w 00$1 $2); do
    DATASET=${i}/
    d=${ML1PATH}${DATASET}
    RESULTS=${TRAIN_DATA}${DATASET}

    echo "Training energy regressor ${DXY} for dataset ${d}"
    python train_energy_regressor.py -e ${d}train_events.csv -t ${d}train_telescopes.csv -H ${RESULTS}hillas.csv -r ${RESULTS}results.csv -o ${TRAIN_DATA}energy_regressor.pickle
done
