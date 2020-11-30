# Setup Instructions for running in NLHPC

NLHPC use a module manager to load modules like Anaconda or CUDA. 

For this we use the command `ml <module>`.

## Creating conda enviroment

First load the modules:

```bash
ml CUDA/10.1.105
ml cuDNN/7.4.2.24
ml Miniconda3
```

Create new enviroment in user level directory and install dependencies.

```bash
conda create --prefix ./envs/gerumo python=3.7
source activate /home/sborquez/envs/gerumo
conda install tensorflow-gpu numpy scipy matplotlib pillow pandas scikit-learn scikit-image astropy pytables seaborn pydot
conda install -c conda-forge colour tqdm
```
## Running Scripts

Use the script generator from NLHPC ([here](https://wiki.nlhpc.cl/Generador_Scripts))

Add this generated script at the begining of the sh file.

```bash
#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -J test
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=4000
#SBATCH --mail-user=sebastian.borquez.g@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o test_%j.out
#SBATCH -e test_%j.err

#-----------------Toolchain---------------------------
ml purge
ml fosscuda/2019b
# ----------------MÃ³dulos-----------------------------
ml  CUDA/10.1.105 HDF5/1.10.4   Miniconda3/4.5.12   cuDNN/7.4.2.24  
# ----------------Comandos--------------------------

source activate /home/sborquez/envs/gerumo
```

Send jobs to queue with

```
sbatch <job>.sh
```

*Note*: never send a job with an enviroment activated!