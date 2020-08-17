# Setup Instructions for running in CCTVAL HPC

CCTVAL HPC use a anaconda

## Creating conda enviroment

First load Anaconda:

```
source /opt/software/anaconda3/2019.03/setup.sh
```

Create new enviroment in user level directory and install dependencies.

```bash
conda create --prefix ./envs/gerumo python=3.7
conda activate /user/s/sborquez/envs/gerumo
conda install tensorflow-gpu numpy scipy matplotlib pillow pandas scikit-learn scikit-image astropy pytables seaborn pydot
conda install -c conda-forge colour tqdm
conda install -c cta-observatory  ctapipe
pip install ctaplot
```
## Running Scripts

Use this template to generate your scripts.

Add this generated script at the begining of the sh file.

```bash
#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -p gpuk <you can use gpum too, but it`s slower>
#SBATCH -J <job name here>
#SBATCH --mail-user=<your email here>
#SBATCH --mail-type=ALL
#SBATCH -o output_<your log name>_%j.log
#SBATCH -e error_<your log name>_%j.log
#SBATCH --gres=gpu:1

# ----------------M�~C³dulos-----------------------------
cd $SLURM_SUBMIT_DIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate <path to the enviroment>/envs/gerumo
```

Send jobs to queue with

```
sbatch <job>.sh
```

*Note*: never send a job with an enviroment activated!


List jobs

```
squeue
```

List nodes

```
sinfo
```




