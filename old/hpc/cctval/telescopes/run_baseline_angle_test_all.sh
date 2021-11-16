#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_all_baseline_test
#SBATCH --mail-user=patricio.campana@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o output_baseline_%j.log
#SBATCH -e error_baseline_%j.log
#SBATCH --time=20:00:00

# ----------------MÃ³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/c/campana/envs/gerumo
echo "Running run_baseline_angle_test_all.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/
DATASET=010/

d=${ML1PATH}${DATASET}
RESULTS=/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/HILLAS/
TEST_RESULTS=${RESULTS}all/

echo "Copying dataset from ${d} to ${TMPDIR}"
cp ${d}*.h5 ${TMPDIR}
cp ${d}events.csv ${TMPDIR}
cp ${d}telescopes.csv ${TMPDIR}

echo "Running angle baseline ${d}"
mkdir -p ${TEST_RESULTS}
python run_baseline.py -e ${TMPDIR}/events.csv -t ${TMPDIR}/telescopes.csv -c ${TEST_RESULTS}hillas.csv -o ${TEST_RESULTS}results.csv -f ${TMPDIR} # -r ${ML1PATH}001/energy_regressor.pickle
