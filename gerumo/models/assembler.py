"""
Multi Observer Assembler
===========
Base models interface for create new multiobservers regresors.
"""

from . import CUSTOM_OBJECTS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from sklearn.metrics import r2_score
import numpy as np

class ModelAssembler():
    def __init__(self, sst1m_model_path=None, mst_model_path=None, lst_model_path=None,
                    targets=[], output_shape=()):
        
        self.models = {}
        self.telescopes = []
        self.sst1m_model_path = sst1m_model_path
        if self.sst1m_model_path is not None:
            self.models["SST1M_DigiCam"] = load_model(self.sst1m_model_path, custom_objects=CUSTOM_OBJECTS)
            self.telescopes.append("SST1M_DigiCam")
            
        self.mst_model_path = mst_model_path
        if self.mst_model_path is not None:
            self.models["MST_FlashCam"] = load_model(self.mst_model_path, custom_objects=CUSTOM_OBJECTS)
            self.telescopes.append("MST_FlashCam")
        
        self.lst_model_path = lst_model_path
        if self.lst_model_path is not None:
            self.__models["LST_LSTCam"] = load_model(self.lst_model_path, custom_objects=CUSTOM_OBJECTS)
            self.telescopes.append("LST_LSTCam")

        self.targets = targets
        self.output_shape = output_shape

    def predict(self, x, **kwargs):
        y_predictions = []
        if isinstance(x, list):
            for x_i in tqdm(x):
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
            for x_batch_j, _ in x:
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

    def evaluate(self, x, y_true_points=None, **kwargs):
        if isinstance(x, list):
            raise y_true_points is None
            y_predictions = self.predict_point(self, x, **kwargs)
        elif isinstance(x, Sequence):
            y_true_points = []
            original_target_mode = x.target_mode
            x.target_mode = "lineal"
            for x_batch_j, y_batch_j in x:
                for x_i, y_i_true in zip(x_batch_j, y_batch_j):
                    y_true_points.append(y_i_true)
                    y_i_by_telescope = {}
                    for telescope in self.telescopes:
                        x_i_telescope = x_i[telescope]
                        # check if events has at least one observation with this telescope type
                        if len(x_i_telescope[0]) > 0:
                            model_telescope = self.models[telescope]
                            y_i_by_telescope[telescope] = model_telescope.predict(x_i_telescope, verbose=0, **kwargs)
                    y_i_assembled = self.assemble(y_i_by_telescope)
                    y_predictions.append(y_i_assembled)
            y_predictions = self.point_estimation(y_predictions)
            x.target_mode = original_target_mode
        
        y_pred = np.array(y_predictions)
        y_true = np.array(y_true_points)

        return r2_score(y_true, y_pred)

    def point_estimation(self, y_predictions):
        raise NotImplementedError

    def assemble(self, y_i_by_telescope):
        raise NotImplementedError


