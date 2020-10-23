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
src_events_csv = "/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/events.csv"
src_telescopes_csv = "/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/telescopes.csv"

# for tel in ["lst", "mst", "sst"]:
#     # filtered datasets (Paths)
#     hillas_csv = f"/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/HILLAS/{tel}/hillas.csv"
#     # dst dataset 
#     dst_events_csv = f"/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/hillas_{tel}_events.csv" 
#     dst_telescopes_csv = f"/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/hillas_{tel}_telescopes.csv" 
#     hillas_filter(src_events_csv, src_telescopes_csv, hillas_csv, dst_events_csv, dst_telescopes_csv)

# filtered datasets (Paths)
hillas_csv = f"/data/atlas/dbetalhc/cta-test/gerumo/output/alt_az/baseline/HILLAS/all/hillas.csv"
# dst dataset 
dst_events_csv = f"/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/hillas_all_events.csv" 
dst_telescopes_csv = f"/data/atlas/dbetalhc/cta-test/gerumo/data/baseline/data_test_10_csv/hillas_all_telescopes.csv" 
hillas_filter(src_events_csv, src_telescopes_csv, hillas_csv, dst_events_csv, dst_telescopes_csv)
