import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')


import os
import json
import argparse

from sklearn.model_selection import train_test_split
from gerumo.baseline.energy import EnergyModel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the baseline energy regressor.")
    ap.add_argument("-e", "--events", type=str, help="Events CSV path used for training", required=True)
    ap.add_argument("-t", "--telescopes", type=str, help="Telescopes CSV path used for training", required=True)
    ap.add_argument("-r", "--results", type=str, help="Hillas reconstruction CSV", required=True)
    ap.add_argument("-H", "--hillas", type=str, help="Hillas parameters CSV", required=True)
    ap.add_argument("-o", "--output", type=str, help="File path where to save the regressor", required=True)

    args = ap.parse_args()

    print("Preparing dataset...")
    dataset = EnergyModel.prepare_dataset(args.events, args.telescopes, args.results, args.hillas)

    regressor = EnergyModel()

    print(f"Training regressor... ({len(dataset)} observations)")
    regressor.fit(dataset)

    print("Saving regressor...")
    regressor.save(args.output)
