#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_test_lst
#SBATCH --mail-user=patricio.campana@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_baseline_%j.log
#SBATCH -e error_baseline_%j.log
#SBATCH --time=48:00:00

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/c/campana/envs/gerumo
echo "Running run_baseline_angle_energy_test_all.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/
DATASET=010/

d=${ML1PATH}${DATASET}
RESULTS=/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/HILLAS/energy/
TRAIN_RESULTS=${RESULTS}train/tight/
TEST_RESULTS=${RESULTS}test/tight/lst/

echo "Copying dataset from ${d} to ${TMPDIR}"
cp ${d}*.h5 ${TMPDIR}
cp ${d}events.csv ${TMPDIR}
cp ${d}telescopes.csv ${TMPDIR}

echo "Running angle baseline ${d}"
mkdir -p ${TEST_RESULTS}
python run_baseline.py -e ${TMPDIR}/events.csv -t ${TMPDIR}/telescopes.csv -c ${TEST_RESULTS}hillas.csv -o ${TEST_RESULTS}results.csv -f ${TMPDIR} -r ${TRAIN_RESULTS}energy_regressor.pickle -T LST_LSTCam
