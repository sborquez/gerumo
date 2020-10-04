#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_validation
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
RESULTS=${d}baseline/
DXY=${ML1PATH}baseline/D$1$2/

echo "Copying energy regressor from ${DXY} to ${TMPDIR}"
cp ${DXY}energy_regressor.pickle ${TMPDIR}/energy_regressor.pickle

for i in {00$1..$2} do
    d=${ML1PATH}$i/

    echo "Copying dataset from ${d} to ${TMPDIR}"
    cp ${d}*.h5 ${TMPDIR}
    cp ${d}validation_events.csv ${TMPDIR}
    cp ${d}validation_telescopes.csv ${TMPDIR}

    echo "Running angle baseline ${d}"
    VAL_RESULTS=${RESULTS}validation/D$1$2/$i/
    mkdir -p ${VAL_RESULTS}
    python run_baseline.py -e ${TMPDIR}/validation_events.csv -t ${TMPDIR}/validation_telescopes.csv -c ${VAL_RESULTS}hillas.csv -o ${VAL_RESULTS}results.csv -n 9999999 -P ${VAL_RESULTS}resolutions.png -f ${TMPDIR} -r ${TMPDIR}/energy_regressor.pickle
