#FIX: this. re structuring the project
import sys
sys.path.insert(1, '..')

import os
import argparse
import numpy as np

import glob
import ctaplot
import pandas as pd
import astropy.units as u

import matplotlib.pyplot as plt

from gerumo.baseline.energy import EnergyModel
from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--experiment", type=str, help="The DXY experiment folder", required=True)
    ap.add_argument("-s", "--save_path", type=str, help="Path where the angular/energy resolution plot will be saved", required=True)
    args = ap.parse_args()

    all_results = list()
    for path in glob.glob(os.path.join(args.experiment, "*/results.csv")):
        all_results.append(pd.read_csv(path, delimiter=","))
    
    results = pd.concat(all_results)

    reco_alt = results['pred_alt']
    reco_az = results['pred_az']

    alt = results['alt']
    az = results['az']

    energy = results['energy']
    mc_energy = results['mc_energy']

    Reconstructor.plot(results, args.save_path)
