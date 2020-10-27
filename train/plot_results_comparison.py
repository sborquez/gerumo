#FIX: this. re structuring the project
import sys
sys.path.insert(1, '..')

import os
import argparse

import pandas

from glob import glob

from gerumo.baseline.energy import EnergyModel
from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--results", type=str, help="Results CSV files (script will search results.csv recursively)", required=True)
    ap.add_argument("-p", "--plot_path", type=str, help="Where to save the plots", required=True)
    # ap.add_argument("-o", "--output", type=str, default=None)

    args = ap.parse_args()

    results = dict()
    labels = {
        "sst": "SST1M_DigiCam",
        "mst": "MST_FlashCam",
        "lst": "LST_LSTCam",
        "all": "All events"
    }
    for file in glob(os.path.join(args.results, "**/results.csv"), recursive=True):
        r = pandas.read_csv(file)
        results[labels[os.path.basename(os.path.dirname(file))]] = r
    
    Reconstructor.plot_comparison(results, save_to=args.plot_path)
