#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#SBATCH -p batch
#SBATCH -J gerumo_baseline_test
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
echo "Running run_baseline_angle_test.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv

d=${ML1PATH}
RESULTS=${ML1PATH}baseline/
TEST_RESULTS=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/baseline/test/sst/

echo "Running angle baseline ${d}"
mkdir -p ${TEST_RESULTS}
python run_baseline.py -e ${ML1PATH}/events.csv -t ${ML1PATH}/telescopes.csv -T SST1M_DigiCam -c ${TEST_RESULTS}hillas.csv -o ${TEST_RESULTS}results.csv -f /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/baseline # -r ${ML1PATH}001/energy_regressor.pickle
