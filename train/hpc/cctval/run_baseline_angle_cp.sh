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
DATASET=002/

d=${ML1PATH}${DATASET}

echo "Copying dataset from ${d} to ${TMPDIR}"
cp ${d}*.h5 ${TMPDIR}
cp ${d}events.csv ${TMPDIR}
cp ${d}telescopes.csv ${TMPDIR}

echo "Run baseline ${d}"
python run_baseline.py -e ${TMPDIR}/events.csv -t ${TMPDIR}/telescopes.csv -c ${d}hillas.csv -o ${d}results.csv -n 9999999 -P ${d}resolutions.png -r ${ML1PATH}001/energy_regressor.pickle -f ${TMPDIR}

