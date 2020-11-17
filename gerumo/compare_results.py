from gerumo import *
import pandas as pd

#todo mode to viz
def compare_results(model_names, csv_files, mode="angular", ylim=(0, 2), xlim=None, save_to=None):
    renamer = {
        'az' : 'true_az',
        'alt' : 'true_alt',
        'mc_energy' : 'true_mc_energy',
        'event_unique_id' : 'event_id',
    }
    results = {}
    for model, csv_file in zip(model_names, csv_files):
        results[model] =  pd.read_csv(csv_file).rename(renamer, axis=1)
    if mode == "angular":
        plot_angular_resolution_comparison(results, ylim=ylim, xlim=xlim, save_to=save_to)
    elif mode == "energy":
        plot_energy_resolution_comparison(results, ylim=ylim, xlim=xlim, save_to=save_to)


if __name__ == "__main__":
    pass