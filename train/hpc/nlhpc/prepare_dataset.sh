#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -J preprocessing
#SBATCH -p slims
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 1
#SBATCH --mem=48000
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o preprocessing_%j.out
#SBATCH -e preprocessing_%j.err

#-----------------Toolchain---------------------------
ml purge
ml intel/2019b
# ----------------MÃ³dulos-----------------------------
ml  CUDA/10.1.105 HDF5/1.10.4   Miniconda3/4.5.12   cuDNN/7.4.2.24  
# ----------------Comandos--------------------------

source activate /home/sborquez/envs/gerumo

#conda list

echo "Running prepare_dataset.sh"
echo ""

cd /home/sborquez/gerumo/train

# Train small
## ls -d  ~/data/train/* | head -n 30
echo "Preprocessing Small"
python prepare_dataset.py -o  /home/sborquez/data/small -I /home/sborquez/data/train/gamma_20deg_0deg_runs100-1155___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs101-1153___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs10-1152___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs102-1675___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs103-1049___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs105-1209___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs106-1260___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs107-1596___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs108-1411___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs111-1520___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs113-1465___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs1132-1577___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs114-1551___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs1157-1611___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs116-1376___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs1-1692___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs119-1059___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs11-958___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs120-1618___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs121-949___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs123-1694___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs126-1672___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs127-1583___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs1276-1602___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs128-1161___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs130-1259___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs131-1320___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs13-1317___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs132-1516___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5 /home/sborquez/data/train/gamma_20deg_0deg_runs133-1482___cta-prod3-demo_desert-2150m-Paranal-baseline_cone10.h5

# Train all
echo "Preprocessing All"
python prepare_dataset.py -o  /home/sborquez/data/train -i /home/sborquez/data/train

# Test
echo "Preprocessing Test"
python prepare_dataset.py -o  /home/sborquez/data/test -i /home/sborquez/data/test -s 0 
