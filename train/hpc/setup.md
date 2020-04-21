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
```
## Running Scripts

Use this template to generate your scripts.

Add this generated script at the begining of the sh file.

```bash
#!/bin/bash
# ----------------SLURM Parameters----------------
#!/bin/bash
#PBS -q gpuk
#PBS -N test
#PBS -l walltime=168:00:00

# ----------------MÃ³dulos-----------------------------
cd $PBS_O_WORKDIR
source /opt/software/anaconda3/2019.03/setup.sh
# ----------------Comandos--------------------------

source activate /user/s/sborquez/envs/gerumo
```

Send jobs to queue with

```
qsub <job>.sh
```

*Note*: never send a job with an enviroment activated!