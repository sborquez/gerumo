import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')


from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    events_path = "../dataset/events.csv"
    telescopes_path = "../dataset/telescopes.csv"
    reco = Reconstructor(events_path, telescopes_path)
    reco.plot_metrics(max_events=1000, min_valid_observations=8)
