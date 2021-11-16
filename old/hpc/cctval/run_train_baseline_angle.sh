#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_train
#SBATCH --mail-user=patricio.campana@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_baseline_%j.log
#SBATCH -e error_baseline_%j.log
#SBATCH --time=30:00:00

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/c/campana/envs/gerumo
echo "Running run_baseline.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/
DATASET=$1/

d=${ML1PATH}${DATASET}
RESULTS=${d}baseline/
TRAIN_RESULTS=${RESULTS}train/

echo "Copying dataset from ${d} to ${TMPDIR}"
cp ${d}*.h5 ${TMPDIR}
cp ${d}train_events.csv ${TMPDIR}
cp ${d}train_telescopes.csv ${TMPDIR}

echo "Running angle baseline ${d} for training set"
mkdir ${RESULTS}
mkdir ${TRAIN_RESULTS}
python run_baseline.py -e ${TMPDIR}/train_events.csv -t ${TMPDIR}/train_telescopes.csv -c ${TRAIN_RESULTS}hillas.csv -o ${TRAIN_RESULTS}results.csv -n 9999999 -P ${TRAIN_RESULTS}resolutions.png -f ${TMPDIR} # -r ${ML1PATH}001/energy_regressor.pickle
