#FIX: this. re structuring the project
import sys

sys.path.insert(1, '..')

import os
import argparse

import pandas
import numpy as np

from glob import glob


from gerumo.visualization.metrics import plot_energy_resolution_comparison, plot_angular_resolution_comparison
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
        "assembler": "All telescopes",
        "all": "All telescopes"
    }
    for file in glob(os.path.join(args.results, "**/results.csv"), recursive=True):
        r = pandas.read_csv(file)
        r["true_alt"] = r["alt"]
        r["true_az"] = r["az"]
        r["true_mc_energy"] = r["mc_energy"]
        r["pred_log10_mc_energy"] = np.log10(r["energy"])
        results[labels[os.path.basename(os.path.dirname(file))]] = r

    plot_angular_resolution_comparison(results, ylim=[0, 2],
                                       save_to=os.path.join(args.plot_path, "angular_resolution_comparable.png"))
    plot_energy_resolution_comparison(results, ylim=[0, 2],
                                      save_to=os.path.join(args.plot_path, "energy_resolution_comparable.png"))
