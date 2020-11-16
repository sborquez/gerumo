import pandas as pd


def hillas_filter(src_events_csv, src_telescopes_csv, hillas_csv, dst_events_csv, dst_telescopes_csv):
    # Load
    events = pd.read_csv(src_events_csv, delimiter=";")
    telescopes = pd.read_csv(src_telescopes_csv, delimiter=";")
    hillas = pd.read_csv(hillas_csv)

    # Filter
    hillas_events = hillas.event_unique_id.unique()
    hillas_telescopes_types = hillas.type.unique()

    events_filtered = events[events["event_unique_id"].isin(hillas_events)]
    telescopes_filtered = hillas.merge(telescopes, on=["event_unique_id", "type", "observation_indice"], validate="1:1", suffixes=('_x', ''))
    telescopes_filtered = telescopes_filtered.drop(columns=["width", "length","phi", "psi", "r", "skewness", "intensity", "kurtosis", "y_x", "x_x"])

    # Save
    events_filtered.to_csv(dst_events_csv, sep=";", index=False)
    telescopes_filtered.to_csv(dst_telescopes_csv, sep=";", index=False)

# source dataset (Boris)
src_events_csv = "/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/full/events.csv"
src_telescopes_csv = "/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/full/telescopes.csv"
# filter
hillas_base = "/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/HILLAS/nocuts/"
# dest
dst_base    = "/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/loose" 

# # filtered datasets (Paths)
for tel in ["all", "sst", "mst","lst"]:
    # filtered datasets (Paths)
    hillas_csv = f"{hillas_base}/{tel}/hillas.csv"
    # dst dataset 
    dst_events_csv = f"{dst_base}/{tel}_events.csv" 
    dst_telescopes_csv = f"{dst_base}/{tel}_telescopes.csv" 
    hillas_filter(src_events_csv, src_telescopes_csv, hillas_csv, dst_events_csv, dst_telescopes_csv)
