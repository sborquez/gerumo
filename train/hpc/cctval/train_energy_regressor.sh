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

echo "Run baseline 001"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 002"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 003"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 004"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 005"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 006"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 007"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 008"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 009"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009/energy_regressor.pickle -n 999999 -v ML1

echo "Run baseline 010"
python train_energy_regressor.py -e /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010/events.csv -t /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010/telescopes.csv -H /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010/hillas.csv -r /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010/results_hillas.csv -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010/energy_regressor.pickle -n 999999 -v ML1
