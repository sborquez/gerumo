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
echo "Running run_baseline.sh"
echo ""

cd /user/c/campana/gerumo/train
ML1PATH=/data/atlas/dbetalhc/cta-test/ML1_SAMPLES/

for d in ${ML1PATH}*/; do
    echo "Run baseline ${d}"
    python run_baseline.py -e ${d}events.csv -t ${d}telescopes.csv -c ${d}hillas.csv -o ${d}results_hillas.csv -n 9999999 -P ${d}angular_res.png -v ML1
done
