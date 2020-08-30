#FIX: this. re structuring the project
import sys
sys.path.insert(1, '..')

import argparse

from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_events", help="Maximum number of events to use", default=100, type=int)
    ap.add_argument("-e", "--events_path", type=str, default="../dataset/events.csv",
                    help="Events folder")
    ap.add_argument("-t", "--telescopes_path", type=str, default="../dataset/telescopes.csv",
                    help="Telescopes folder")
    ap.add_argument("-p", "--plot", action="store_true", help="Plot charges images")
    ap.add_argument("-o", "--output", type=str, default=None)
    ap.add_argument("-c", "--hillas_csv", type=str, default=None)
    args = ap.parse_args()

    reco = Reconstructor(args.events_path, args.telescopes_path)
    reco.plot_metrics(max_events=args.n_events, min_valid_observations=2, plot_charges=args.plot, save_to=args.output, save_hillas=args.hillas_csv)
