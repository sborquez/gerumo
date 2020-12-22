from os import path
import pandas as pd
import numpy as np
import os
from .metrics import plot_model_validation_regressions, plot_prediction
from .explore import plot_input_sample
from tensorflow import keras


__all__ = [
  'ValidationRegressionCallback', 'ValidationSamplesCallback'
]

# Callbacks
class ValidationRegressionCallback(keras.callbacks.Callback):
    def __init__(self, validation_generator, regressions_folder, assembler, save_best_only=True, mode='min'):
        self.validation_generator = validation_generator
        self.assembler = assembler
        self.targets = validation_generator.targets
        self.regressions_folder = regressions_folder
        self.regressions_image_fmt = path.join(regressions_folder, "epoch_{epoch}_loss_{val_loss:.4f}.png")
        self.save_best_only = save_best_only
        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda new_value, best_value: (new_value < best_value)
        elif mode == "max":
            self.best_value = float("-inf")
            self.is_better = lambda new_value, best_value: (new_value > best_value)
        else:
            raise ValueError(f"invalid mode: {mode}")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if (not self.save_best_only) or (self.save_best_only and self.is_better(val_loss, self.best_value)):
            self.best_value = val_loss
            regressions_image_filepath = self.regressions_image_fmt.format(epoch=epoch, val_loss=val_loss)
            print("Saving regression plot in", regressions_image_filepath)
            # Make predictions and save each plot 
            predictions_points = []
            targets_values = []
            # TODO: Add metadata and angular resolution
            original_target_mode = self.validation_generator.target_mode
            self.validation_generator.target_mode = 'lineal'
            for batch_x, batch_t in self.validation_generator:
                targets_values.extend(batch_t)
                batch_p = self.assembler.exec_model_estimation(batch_x, self.model, 0)
                batch_p_points = self.assembler.point_estimation(batch_p)
                predictions_points.extend(batch_p_points)
            self.validation_generator.target_mode = original_target_mode
            evaluation_results = {
                #"event_id":         results["event_id"],       # not used in regression plot
                #"telescope_id":     results["telescope_id"],   # not used in regression plot
                #"telescope_type":   results["telescope_type"], # not used in regression plot
                # "true_mc_energy":   results["true_energy"],   # not used in regression plot
            }
            for target_i, target_name in enumerate(self.targets):
                evaluation_results[f"true_{target_name}"] = np.array(targets_values)[:,target_i].flatten()
                evaluation_results[f"pred_{target_name}"] = np.array(predictions_points)[:,target_i].flatten()
            evaluation_results = pd.DataFrame(evaluation_results).dropna()
            plot_model_validation_regressions(evaluation_results, self.targets, save_to=regressions_image_filepath)
        

class ValidationSamplesCallback(keras.callbacks.Callback):

    def __init__(self, sample_generator, predictions_folder, assembler, save_best_only=True, mode='min'):
        assert sample_generator.include_event_id is True, "sample_generator doesn`t include event_id"
        assert sample_generator.include_true_energy is True, "sample_generator doesn`t include true_energy"
        assert sample_generator.target_mode == "lineal", "sample_generator target_mode is not 'lineal'"
        # Generator
        self.sample_generator = sample_generator
        # Assembler
        self.assembler = assembler
        # Input Image
        self.input_image_mode = self.sample_generator.input_image_mode
        # Target
        self.targets = sample_generator.targets
        self.target_mode = sample_generator.target_mode
        self.target_domains = assembler.target_domains
        self.target_resolutions = self.assembler.target_resolutions
        # Predictions paths
        self.predictions_folder = predictions_folder
        self.predictions_subfolder_fmt = path.join(predictions_folder, "epoch_{epoch}")
        self.save_best_only = save_best_only
        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda new_value, best_value: (new_value < best_value)
        elif mode == "max":
            self.best_value = float("-inf")
            self.is_better = lambda new_value, best_value: (new_value > best_value)
        else:
            raise ValueError(f"invalid mode: {mode}")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if (not self.save_best_only) or (self.save_best_only and self.is_better(val_loss, self.best_value)):
            self.best_value = val_loss
            predictions_subfolder = self.predictions_subfolder_fmt.format(epoch=epoch)
            os.makedirs(predictions_subfolder, exist_ok=False)
            print("Saving predictions in", predictions_subfolder)
            # Make predictions and save each plot 
            for batch_x, batch_t, meta in self.sample_generator:
                # Predictions
                batch_p = self.assembler.exec_model_estimation(batch_x, self.model, 0) # TODO: ineficiente con BMO
                batch_p_points = self.assembler.point_estimation(batch_p)
                batch_input_image, batch_input_features = batch_x
                # save  prediction
                batch_iterator = zip(batch_input_image, batch_input_features, batch_p, batch_p_points, batch_t, meta["event_id"], meta["telescope_id"])
                for input_image_sample, input_features_sample, prediction, prediction_point, target_sample, event_id, telescope_id in batch_iterator:
                    image_filepath = path.join(predictions_subfolder, f"event_id_{event_id}_telescope_id_{telescope_id}_input.png")
                    plot_input_sample(input_image_sample, self.input_image_mode, input_features_sample, title=(event_id, telescope_id), make_simple=True,
                                       save_to=image_filepath)
                    
                    prediction_filepath = path.join(predictions_subfolder, f"event_id_{event_id}_telescope_id_{telescope_id}_prediction.png")
                    plot_prediction(prediction, prediction_point, self.targets,
                                     self.target_domains, self.target_resolutions, 
                                    (event_id, telescope_id), target_sample, save_to=prediction_filepath)

