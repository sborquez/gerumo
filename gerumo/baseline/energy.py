import _pickle as pickle

import tqdm
import numpy as np

from pandas import DataFrame

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from ctapipe.utils import CutFlow
from ctapipe.containers import HillasParametersContainer

from gerumo import load_camera
from gerumo.baseline.io import load_hillas_dataset, aggregate_hillas_dataset
from gerumo.baseline.mapper import split_tel_type
from gerumo.baseline.cutflow import generate_observation_cutflow


class EnergyModel:
    def __init__(self):
        self._models = dict()
        self._features = ["log10_intensity", "log10_impact", "width", "length", "h_max"]

    @staticmethod
    def prepare_dataset(events_csv: str, telescopes_csv: str, results_csv: str, hillas_csv: str, split: float = None):
        dataset = load_hillas_dataset(events_csv, telescopes_csv, results_csv, hillas_csv)
        event_ids = dataset["event_unique_id"].unique()
        np.random.shuffle(event_ids)

        if split is None:
            return aggregate_hillas_dataset(dataset)
        
        n_test = int(np.ceil(split * len(event_ids)))
        test = dataset[dataset["event_unique_id"].isin(event_ids[:n_test])]
        train = dataset[dataset["event_unique_id"].isin(event_ids[n_test:])]

        return aggregate_hillas_dataset(train), aggregate_hillas_dataset(test)

    def fit(self, dataset, param_grid = None, cv = 5, scoring = "neg_mean_squared_error"):
        if param_grid is None:
            param_grid = {
                "n_estimators": np.linspace(100, 1000, 50, dtype=int)
            }

        grouped = dataset[["type", "mc_energy"] + self._features].groupby("type")
        for t, group in grouped:
            tel_type = t.split("_")[1]
            if tel_type not in self._models:
                self._models[tel_type] = GridSearchCV(
                    estimator=RandomForestRegressor(max_depth=None),
                    param_grid=param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    verbose=2
                )
            x = group[self._features].values
            y = group["mc_energy"].values
            self._models[tel_type].fit(x, y)

    def predict_dataset(self, dataset):
        grouped = dataset[["event_unique_id", "observation_indice", "type", "intensity", "mc_energy"] + self._features].groupby("event_unique_id")
        
        results = {}
        for event_id, group in grouped:
            energies = np.zeros(len(group))
            weights = np.zeros(len(group))

            count = 0
            for _, row in group.set_index("observation_indice").iterrows():
                x = row[self._features]
                weights[count] = row["intensity"]
                energies[count] = self._models[row["type"]].predict(x.values.reshape(1, -1))
                count += 1

            pred_energy = np.sum(weights * energies) / np.sum(weights)
            mc_energy = group["mc_energy"].unique()[0]

            results[event_id] = {
                "pred": pred_energy,
                "mc": mc_energy
            }

        return results

    def predict_event(self, positions, types, hillas_containers, reconstruction):
        n_obs = len(hillas_containers)
        energies = np.zeros(n_obs)
        weights = np.zeros(n_obs)

        data = {tel_id: (hillas_containers[tel_id], positions[tel_id], types[tel_id]) for tel_id in positions}
        for idx, (moments, position, t) in enumerate(data.values()):
            x, y = position
            impact = np.sqrt((reconstruction.core_x.value - x) ** 2 + (reconstruction.core_y.value - y) ** 2)
            
            X = np.array([[
                np.log10(moments.intensity),
                np.log10(impact),
                moments.width.value,
                moments.length.value,
                reconstruction.h_max.value
            ]])
            weights[idx] = moments.intensity
            
            tel_type = t.split("_")[1]
            energies[idx] = self._models[tel_type].predict(X)

        pred_energy = np.sum(weights * energies) / np.sum(weights)
        return pred_energy

    def save(self, path: str):
        dumped = pickle.dumps(self)
        with open(path, "wb") as f:
            f.write(dumped)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
