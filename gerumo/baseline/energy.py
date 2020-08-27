import pickle

import numpy as np
import tqdm

from pandas import DataFrame

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from ctapipe.utils import CutFlow
# from ctapipe.reco import EnergyRegressor
from ctapipe.containers import HillasParametersContainer

from gerumo import load_camera
from gerumo.baseline.mapper import split_tel_type
from gerumo.baseline.cutflow import generate_observation_cutflow
from gerumo.baseline.reconstructor import get_camera_radius, get_observation_parameters


def fill_row_hillas_params(row, cutflow: CutFlow, version="ML1", loading_bar: tqdm.tqdm = None):
    fields = ["source", "folder", "observation_indice", "type"]
    source, folder, tel_id, tel_type = row[fields]
    print(source, folder, tel_id, tel_type)
    charge, peak = load_camera(source=source, folder=folder, telescope_type=tel_type, observation_indice=tel_id,
                               version=version)

    _, camera_name = split_tel_type(tel_type)

    params = get_observation_parameters(charge, peak, camera_name, cutflow)
    if params is None:
        row["hillas_width"] = None
        row["hillas_length"] = None
        row["hillas_skewness"] = None
        row["hillas_kurtosis"] = None
        return row

    moments, _, _, _ = params
    moments: HillasParametersContainer

    row["hillas_width"] = moments.width.value
    row["hillas_length"] = moments.length.value
    row["hillas_skewness"] = moments.skewness
    row["hillas_kurtosis"] = moments.kurtosis

    if loading_bar:
        loading_bar.update()
    return row


def aggregate_dataset_hillas(dataset: DataFrame, version="ML1"):
    camera_radius = get_camera_radius(dataset)
    obs_cutflow = generate_observation_cutflow(camera_radius)

    bar = tqdm.tqdm(total=len(dataset))
    print(dataset.head())
    dataset = dataset.apply(lambda row: fill_row_hillas_params(row=row, cutflow=obs_cutflow, version=version,
                                                               loading_bar=bar), axis=1)
    bar.close()
    return dataset


def filter_dataset_hillas(dataset: DataFrame):
    pass


class EnergyModel:
    def __init__(self):
        self._model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None))

    @staticmethod
    def prepare_dataset(dataset):
        return aggregate_dataset_hillas(dataset)

    def fit(self, train_dataset, validation_dataset):
        pass

    def predict(self, width, length, skewness, kurtosis, h_max):
        return self._model.predict([np.array(width, length, skewness, kurtosis, h_max)])

    def save(self, path: str):
        with open(path, "w") as f:
            pickle.dump(self._model, f)
