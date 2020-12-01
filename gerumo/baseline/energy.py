import _pickle as pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from gerumo.data.dataset import load_hillas_dataset, aggregate_hillas_dataset


class EnergyModel:
    def __init__(self):
        self._models = dict()
        self._features = [
            "log10_intensity",
            "width",
            "length",
            "x", "y",
            "psi",
            "phi",
            "wl",
            "skewness",
            "kurtosis",
            "r",
            "time_gradient",
            "leakage_intensity_width_2",
            "n_islands"
        ]

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

    def fit(self, dataset, param_grid=None, cv=5, scoring="neg_mean_squared_error"):
        if param_grid is None:
            param_grid = {
                "n_estimators": np.linspace(200, 500, 5, dtype=int),
                "min_samples_split": [2, 6]
            }

        random_forest_regressor_args = {
            "max_depth": 50,
            "min_samples_leaf": 2,
            "n_jobs": 4,
            "n_estimators": 150,
            "bootstrap": True,
            "criterion": "mse",
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "min_impurity_split": None,
            "min_samples_split": 2,
            "min_weight_fraction_leaf": 0.0,
            "oob_score": False,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False
        }

        grouped = dataset[["type", "log10_mc_energy"] + self._features].groupby("type")
        for t, group in grouped:
            tel_type = t.split("_")[1]
            if tel_type not in self._models:
                self._models[tel_type] = RandomForestRegressor(**random_forest_regressor_args)
            x = group[self._features].values
            y = group["log10_mc_energy"].values
            self._models[tel_type].fit(x, y)

    def predict_dataset(self, dataset):
        grouped = dataset[
            ["event_unique_id", "observation_indice", "type", "intensity", "mc_energy"] + self._features].groupby(
            "event_unique_id")

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
            mc_energy = group["log10_mc_energy"].unique()[0]

            results[event_id] = {
                "pred": pred_energy,
                "mc": mc_energy
            }

        return results

    def predict_event(self, positions, types, hillas_containers, reconstruction, time_gradients, leakages, n_islands,
                      meta):
        n_obs = len(hillas_containers)
        energies = np.zeros(n_obs)
        weights = np.zeros(n_obs)

        data = {tel_id: (hillas_containers[tel_id], positions[tel_id], types[tel_id], meta[tel_id]) for tel_id in
                positions}
        for idx, (moments, position, t, (tel_type, obs_id)) in enumerate(data.values()):
            x, y = position

            X = np.array([[
                np.log10(moments.intensity),
                moments.width.value,
                moments.length.value,
                x, y,
                moments.psi.value,
                moments.phi.value,
                moments.width.value / moments.length.value,
                moments.skewness,
                moments.kurtosis,
                moments.r.value,
                time_gradients[(tel_type, obs_id)],
                leakages[(tel_type, obs_id)].intensity_width_2,
                n_islands[(tel_type, obs_id)]
            ]])
            weights[idx] = moments.intensity

            tel_type = t.split("_")[1]
            energies[idx] = self._models[tel_type].predict(X)

        pred_energy = np.sum(weights * np.power(10, energies)) / np.sum(weights)
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
