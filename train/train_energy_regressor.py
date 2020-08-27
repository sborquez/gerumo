import os
import argparse

from gerumo import load_dataset
from gerumo.baseline.energy import EnergyModel

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the baseline energy regressor.")
    ap.add_argument("-e", "--train_events", type=str, help="Events CSV path used for training",
                    default="../dataset/train_events.csv")
    ap.add_argument("-t", "--train_telescopes", type=str, help="Telescopes CSV path used for training",
                    default="../dataset/train_telescopes.csv")
    ap.add_argument("-E", "--val_events", type=str, help="Events CSV path used for validation",
                    default="../dataset/validation_events.csv")
    ap.add_argument("-T", "--val_telescopes", type=str, help="Telescopes CSV path used for validation",
                    default="../dataset/validation_telescopes.csv")

    args = ap.parse_args()

    train_dataset = load_dataset(events_path=args.train_events, telescopes_path=args.train_telescopes)
    val_dataset = load_dataset(events_path=args.val_events, telescopes_path=args.val_telescopes)

    train_dataset = EnergyModel.prepare_dataset(train_dataset)
    # val_dataset = EnergyTrainer.prepare_dataset(val_dataset)

    dirname = os.path.dirname(args.train_events)
    train_dataset.to_csv(os.path.join(dirname, "agg_train.csv"))
