#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_test_mst
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
echo "Running run_baseline.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/
DATASET=010/

d=${ML1PATH}${DATASET}
RESULTS=${d}baseline/
TEST_RESULTS=${RESULTS}test/nocuts/mst/

echo "Copying dataset from ${d} to ${TMPDIR}"
cp ${d}*.h5 ${TMPDIR}
cp ${d}events.csv ${TMPDIR}
cp ${d}telescopes.csv ${TMPDIR}

echo "Running angle baseline ${d}"
mkdir -p ${TEST_RESULTS}
python run_baseline.py -e ${TMPDIR}/events.csv -t ${TMPDIR}/telescopes.csv -T MST_FlashCam -c ${TEST_RESULTS}hillas.csv -o ${TEST_RESULTS}results.csv -f ${TMPDIR} # -r ${ML1PATH}001/energy_regressor.pickle
