#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -p gpuk
#SBATCH -J gerumo_preprocessing
#SBATCH --mail-user=patricio.campana@sansano.usm.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/output_preprocessing_%j.log
#SBATCH -e logs/error_preprocessing_%j.log
#SBATCH --gres=gpu:1

# ----------------Módulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/c/campana/envs/gerumo
echo "Running prepare_dataset.sh"
echo ""

cd /user/c/campana/gerumo/train

echo "Preprocessing 001"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/001 -s 0.2 -v ML1

echo "Preprocessing 002"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/002 -s 0.2 -v ML1

echo "Preprocessing 003"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/003 -s 0.2 -v ML1

echo "Preprocessing 004"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/004 -s 0.2 -v ML1

echo "Preprocessing 005"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/005 -s 0.2 -v ML1

echo "Preprocessing 006"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/006 -s 0.2 -v ML1

echo "Preprocessing 007"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/007 -s 0.2 -v ML1

echo "Preprocessing 008"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/008 -s 0.2 -v ML1

echo "Preprocessing 009"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/009 -s 0.2 -v ML1

echo "Preprocessing 010"
python prepare_dataset.py -o /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010 -i /data/atlas/dbetalhc/cta-test/ML1_SAMPLES/010 -s 0.2 -v ML1
