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

Add this line at the begining of the sh file.

```bash
ml HDF5
ml CUDA/10.1.105
ml cuDNN/7.4.2.24
ml Miniconda3

source activate /home/sborquez/envs/gerumo
```