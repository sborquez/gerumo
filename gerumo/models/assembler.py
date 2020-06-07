"""
Multi Observer Assembler
===========
Base models interface for create new multiobservers regresors.
"""

from . import CUSTOM_OBJECTS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tqdm import tqdm
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

class ModelAssembler():
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, lst_model_or_path=None,
                    targets=[], target_domains={}, target_shapes=(), custom_objects=CUSTOM_OBJECTS):
        assert not ((sst1m_model_or_path is None) and (mst_model_or_path is None) and (lst_model_or_path is None)), "No models given" 
        self.models = {}
        self.telescopes = []

        # LOAD SST1M_DigiCam Model
        if sst1m_model_or_path is not None:
            self.telescopes.append("SST1M_DigiCam")
            if isinstance(sst1m_model_or_path, str):
                self.sst1m_model_path = sst1m_model_or_path
                self.models["SST1M_DigiCam"] = load_model(self.sst1m_model_path, custom_objects=custom_objects)
            elif isinstance(sst1m_model_or_path, Model):
                self.sst1m_model_path = None
                self.models["SST1M_DigiCam"] = sst1m_model_or_path
        else:
            self.sst1m_model_path = None

        # LOAD MST_FlashCam Model
        if mst_model_or_path is not None:
            self.telescopes.append("MST_FlashCam")
            if isinstance(mst_model_or_path, str):
                self.mst_model_path = mst_model_or_path
                self.models["MST_FlashCam"] = load_model(self.mst_model_path, custom_objects=custom_objects)
            elif isinstance(mst_model_or_path, Model):
                self.mst_model_path = None
                self.models["MST_FlashCam"] = mst_model_or_path
        else:
            self.mst_model_path = None

        # LOAD LST_LSTCam Model
        if lst_model_or_path is not None:
            self.telescopes.append("LST_LSTCam")
            if isinstance(lst_model_or_path, str):
                self.lst_model_path = lst_model_or_path
                self.models["LST_LSTCam"] = load_model(self.lst_model_path, custom_objects=custom_objects)
            elif isinstance(lst_model_or_path, Model):
                self.lst_model_path = None
                self.models["LST_LSTCam"] = lst_model_or_path
        else:
            self.lst_model_path = None

        self.targets = targets
        self.target_domains = target_domains
        self.target_shapes = target_shapes
        self.mode = ModelAssembler.select_mode(targets)

    @staticmethod
    def select_mode(targets):
        if len(targets) == 1 and targets[0] in ["mc_energy", "log10_mc_energy"]:
            return "energy reconstruction"

        elif len(targets) == 2 and sorted(targets) == sorted(["alt", "az"]):
            return "angular reconstruction"

        elif len(targets) == 3 and\
            (sorted(targets) == sorted(["alt", "az", "mc_energy"]) \
                or sorted(targets) == sorted(["alt", "az", "log10_mc_energy"])):
            return "complete reconstruction"

    def predict(self, x, pbar=None, **kwargs):
        y_predictions = []
        if isinstance(x, list) or isinstance(x, np.ndarray):
            iterator = x if pbar is None else pbar(x)
            for x_i in iterator:
                y_i_by_telescope = {}
                for telescope in self.telescopes:
                    x_i_telescope = x_i[telescope]
                    # check if events has at least one observation with this telescope type
                    if len(x_i_telescope[0]) > 0: 
                        model_telescope = self.models[telescope]
                        y_i_by_telescope[telescope] = model_telescope.predict(x_i_telescope, verbose=0, **kwargs)
                y_i_assembled = self.assemble(y_i_by_telescope)
                y_predictions.append(y_i_assembled)

        elif isinstance(x, Sequence):
            iterator = x if pbar is None else pbar(x)
            for x_batch_j, _ in iterator:
                for x_i in x_batch_j:
                    y_i_by_telescope = {}
                    for telescope in self.telescopes:
                        x_i_telescope = x_i[telescope]
                        # check if events has at least one observation with this telescope type
                        if len(x_i_telescope[0]) > 0:
                            model_telescope = self.models[telescope]
                            y_i_by_telescope[telescope] = model_telescope.predict(x_i_telescope, verbose=0, **kwargs)
                    y_i_assembled = self.assemble(y_i_by_telescope)
                    y_predictions.append(y_i_assembled)

        return np.array(y_predictions)

    def predict_point(self, x, **kwargs):
        y_predictions = self.predict(x, **kwargs)
        return self.point_estimation(y_predictions)

    def evaluate(self, x, y_true_points=None, return_event_predictions=None, pbar=None, **kwargs):
        """
        evaluate predict points from a generator, return a point predictions table.
        """

        if isinstance(x, list):
            raise NotImplementedError
            # raise y_true_points is None
            # events_ids = [x_i["event_id"] if "event_id" in x_i else i for i, x_i in enumerate(x)]
            # y_predictions = self.predict_point(self, x, **kwargs)

        elif isinstance(x, Sequence):
            # Evaluation data
            y_true_points = []
            event_ids = []
            true_energy = []
            y_predictions = []
            y_predictions_points = []
            
            # Save original generator parameters
            original_target_mode = x.target_mode
            original_event_flag = x.include_event_id
            original_true_energy_flag = x.include_true_energy
            
            # Set evaluation parameters to generator
            x.target_mode = "lineal"
            x.include_event_id = True
            x.include_true_energy = True

            # Iterate for each batch
            iterator = x if pbar is None else pbar(x)
            for x_batch_j, y_batch_j, meta in iterator:
                # Model predictions 
                predictions_batch_j = self.predict(x_batch_j)
                point_predictions_batch_j = self.point_estimation(predictions_batch_j)

                # Update records
                y_true_points.extend(y_batch_j)
                event_ids.extend(meta["event_id"])
                true_energy.extend(meta["true_energy"])

                if return_event_predictions is not None:
                    for prediction_i, prediction_point_i, target_point_i, event_i in zip(predictions_batch_j, point_predictions_batch_j, y_batch_j, meta["event_id"]):
                        if event_i in return_event_predictions:
                                y_predictions.append((prediction_i, prediction_point_i, target_point_i, event_i))
                y_predictions_points.extend(point_predictions_batch_j)
            
            # Set original generator parameters
            x.target_mode = original_target_mode
            x.include_event_id = original_event_flag
            x.include_true_energy = original_true_energy_flag
        
        results = {
            "predictions": np.array(y_predictions_points), 
            "targets": np.array(y_true_points), 
            "true_energy": np.array(true_energy), 
            "event_id": np.array(event_ids)
        }

        if return_event_predictions is not None:
            return results, y_predictions
        else:
            return results

    def point_estimation(self, y_predictions):
        raise NotImplementedError

    def assemble(self, y_i_by_telescope):
        raise NotImplementedError


