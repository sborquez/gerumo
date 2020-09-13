#FIX: this. re structuring the project
import sys
sys.path.insert(1, '..')

import argparse

from gerumo.baseline.energy import EnergyModel
from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_events", help="Maximum number of events to use", default=100, type=int)
    ap.add_argument("-e", "--events_path", type=str, default="../dataset/events.csv",
                    help="Events folder")
    ap.add_argument("-t", "--telescopes_path", type=str, default="../dataset/telescopes.csv",
                    help="Telescopes folder")
    ap.add_argument("-p", "--plot", action="store_true", help="Plot charges images")
    ap.add_argument("-P", "--plot_path", type=str, help="Where to save the angular resolution plot")
    ap.add_argument("-o", "--output", type=str, default=None)
    ap.add_argument("-c", "--hillas_csv", type=str, default=None)
    ap.add_argument("-r", "--energy_regressor", type=str, default=None)
    ap.add_argument("-f", "--replace_folder", type=str, default=None)
    args = ap.parse_args()

    regressor = None
    if args.energy_regressor is not None:
        regressor = EnergyModel.load(args.energy_regressor)

    reco = Reconstructor(args.events_path, args.telescopes_path, replace_folder=args.replace_folder)
    reco.plot_metrics(
        max_events=args.n_events, min_valid_observations=2, plot_charges=args.plot, save_to=args.output,
        save_hillas=args.hillas_csv, energy_regressor=regressor, save_plots=args.plot_path
    )
