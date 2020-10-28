import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *

import logging
import time
import os
from os import path

import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import matplotlib as mpl
mpl.use('Agg')

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


def train_model(model_name, model_constructor, assembler_constructor, model_extra_params, 
            telescope, 
            train_events_csv, train_telescope_csv,
            validation_events_csv, validation_telescope_csv,
            version = "ML1",
            replace_folder_train = None,
            replace_folder_validation = None,
            output_folder = "./output", 
            min_observations = 3,
            input_image_mode = "simple-shift",
            input_image_mask = True,
            input_features = ["x", "y"],
            target_mode = "lineal",
            targets = ["alt", "az", "log10_mc_energy"],
            target_mode_config = {},
            batch_size = 32,
            epochs = 3,
            loss = "crossentropy",
            optimizer = 'sgd',
            optimizer_parameters = {},
            learning_rate = 1e-1,
            preprocessing_parameters = {},
            save_checkpoints = True, save_predictions=True, save_regressions = True, save_loss=True,
            quiet=False, multi_gpu=False, early_stop_patience=None):
    # Assembler
    assembler = assembler_constructor(targets=targets, 
                                      target_shapes=target_mode_config["target_shapes"],
                                      target_domains=target_mode_config["target_domains"], 
                                      target_resolutions=target_mode_config["target_resolutions"])
    target_domains_list = target_mode_config["target_domains"]
    target_domains = {target: target_domain for target, target_domain in zip(targets, target_domains_list)}

    # Prepare datasets
    train_dataset      = load_dataset(train_events_csv, train_telescope_csv, replace_folder_train)
    validation_dataset = load_dataset(validation_events_csv, validation_telescope_csv, replace_folder_validation)

    train_dataset = aggregate_dataset(train_dataset, az=True, log10_mc_energy=True)
    train_dataset = filter_dataset(train_dataset, telescope, min_observations, target_domains)
    
    validation_dataset = aggregate_dataset(validation_dataset, az=True, log10_mc_energy=True, hdf5_file=True)
    validation_dataset = filter_dataset(validation_dataset, telescope, min_observations, target_domains)

    # Preprocessing pipes
    ## input preprocessing
    preprocess_input_pipes = {}
    if "CameraPipe" in preprocessing_parameters:
        camera_parameters = preprocessing_parameters["CameraPipe"]
        camera_pipe = CameraPipe(telescope_type=telescope, version=version, **camera_parameters)
        preprocess_input_pipes['CameraPipe'] = camera_pipe
    if "TelescopeFeaturesPipe" in preprocessing_parameters:
        telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
        telescope_features_pipe = TelescopeFeaturesPipe(telescope_type=telescope, version=version, **telescopefeatures_parameters)
        preprocess_input_pipes['TelescopeFeaturesPipe'] = telescope_features_pipe
    ## output preprocessing
    preprocess_output_pipes = {}

    # Generators
    train_generator =   AssemblerUnitGenerator(
                            train_dataset, batch_size, 
                            input_image_mode=input_image_mode, 
                            input_image_mask=input_image_mask, 
                            input_features=input_features,
                            targets=targets,
                            target_mode=target_mode, 
                            target_mode_config=target_mode_config,
                            preprocess_input_pipes=preprocess_input_pipes,
                            preprocess_output_pipes=preprocess_output_pipes,
                            version=version
                        )
    validation_generator =  AssemblerUnitGenerator(
                                validation_dataset, max(batch_size//4, 1), 
                                input_image_mode=input_image_mode,
                                input_image_mask=input_image_mask, 
                                input_features=input_features,
                                targets=targets,
                                target_mode=target_mode, 
                                target_mode_config=target_mode_config,
                                preprocess_input_pipes=preprocess_input_pipes,
                                preprocess_output_pipes=preprocess_output_pipes,
                                version=version
                            )
    # CallBacks
    callbacks = []
    if save_checkpoints:
        # Checkpoint parameters
        checkpoint_folder = path.join(output_folder, 'checkpoints')
        os.makedirs(checkpoint_folder, exist_ok=False)
        
        checkpoint_filepath = "%s_%s_%s_e{epoch:03d}_{val_loss:.4f}.h5"%(model_name, telescope, loss)
        checkpoint_filepath = path.join(checkpoint_folder, checkpoint_filepath)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', 
                 verbose=1, save_weights_only=False, mode='min', save_best_only=True))
    if save_loss:
        # CSV logger parameters
        csv_logger_filename = path.join(output_folder, 'loss.csv')
        callbacks.append(
            keras.callbacks.CSVLogger(csv_logger_filename)
        )

    if save_predictions:
        # Prediction parameters
        predictions_folder = path.join(output_folder, 'predictions')
        os.makedirs(predictions_folder, exist_ok=False)

        r = np.random.RandomState(42)
        sample_events = r.choice(validation_dataset["event_unique_id"].unique(), size=5, replace=False)
        sample_dataset = validation_dataset[validation_dataset["event_unique_id"].isin(sample_events)]
        sample_generator =  AssemblerUnitGenerator(
                sample_dataset, len(sample_dataset), 
                input_image_mode=input_image_mode,
                input_image_mask=input_image_mask, 
                input_features=input_features,
                targets=targets,
                target_mode='lineal', 
                target_mode_config=target_mode_config,
                preprocess_input_pipes=preprocess_input_pipes,
                preprocess_output_pipes=preprocess_output_pipes,
                include_event_id=True,
                include_true_energy=True,
                version=version
        )
        callbacks.append(
            ValidationSamplesCallback(sample_generator, predictions_folder, assembler)
        )
    if save_regressions:
        # Prediction parameters
        regression_folder = path.join(output_folder, 'regression')
        os.makedirs(regression_folder, exist_ok=False)
      
        callbacks.append(
            ValidationRegressionCallback(validation_generator, regression_folder, assembler)
        )

    if (early_stop_patience is not None) and (early_stop_patience > 0):
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience)
        )

    # Train
    input_img_shape = INPUT_SHAPE[f"{input_image_mode}-mask" if input_image_mask else input_image_mode][telescope]
    input_features_shape = (len(input_features),)
    target_shapes = target_mode_config["target_shapes"]
    
    # Multi GPU
    if multi_gpu:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = model_constructor(telescope, input_image_mode, input_image_mask,
                        input_img_shape, input_features_shape,
                        targets, target_mode, target_shapes,
                        **model_extra_params)
            ## Loss function
            loss = loss if loss.split('_')[-1] == 'loss' else f'{loss}_loss'
            if loss == "crossentropy":
                loss_ = LOSS[loss](dimensions=len(targets))
            elif loss == "distance":
                loss_ = mean_distance_loss(target_shapes)
            else:
                loss_ = LOSS[loss]()
            ## Optimizer
            optimizer_ = OPTIMIZERS[optimizer](
                learning_rate=learning_rate,
                **optimizer_parameters
            )
            model.compile(
                optimizer=optimizer_,
                loss=loss_
            )

    else: 
        model = model_constructor(telescope, input_image_mode, input_image_mask, 
                    input_img_shape, input_features_shape,
                    targets, target_mode, target_shapes, 
                    **model_extra_params)
        ## Loss function
        loss = loss if loss.split('_')[-1] == 'loss' else f'{loss}_loss'
        if loss == "crossentropy":
            loss_ = LOSS[loss](dimensions=len(targets))
        elif loss == "distance":
            loss_ = mean_distance_loss(target_shapes)
        else:
            loss_ = LOSS[loss]()
        ## Optimizer
        optimizer_ = OPTIMIZERS[optimizer](
            learning_rate=learning_rate,
            **optimizer_parameters
        )
        model.compile(
            optimizer=optimizer_,
            loss=loss_
        )
    
    if quiet:
        from contextlib import redirect_stdout
        with open(path.join(output_folder, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()
    else:
        model.summary()
    ## fit
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs = epochs,
        verbose = 2 if quiet else 1,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks = callbacks,
        use_multiprocessing = False,
        workers = 5,
        max_queue_size = 20,
    )
    training_time = (time.time() - start_time)/60.0
    print(f"Training time: {training_time:.3f} [min]")

    plot_model_training_history(history, training_time, model_name, epochs, output_folder)
    
    return model

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Train a model.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Configuration file for model/experiment.")
    ap.add_argument("-q", "--quiet", action='store_true', dest='quiet')
    ap.add_argument("-G", "--multi-gpu", action='store_true', dest='multi_gpu')
    args = vars(ap.parse_args()) 
    config_file = args["config"]
    quiet = args["quiet"]
    multi_gpu = args["multi_gpu"]
    
    print(f"Loading config from: {config_file}")
    with open(config_file) as cfg_file:
        config = json.load(cfg_file)

    # Model
    model_name = config["model_name"]
    model_constructor = MODELS[config["model_constructor"]]
    assembler_constructor = ASSEMBLERS[config["assembler_constructor"]]
    model_extra_params = config["model_extra_params"]

    # Dataset Parameters
    version = config["version"]
    output_folder = config["output_folder"]
    replace_folder_train = config["replace_folder_train"]
    replace_folder_validation = config["replace_folder_validation"]
    train_events_csv    = config["train_events_csv"]
    train_telescope_csv = config["train_telescope_csv"]
    validation_events_csv    = config["validation_events_csv"]
    validation_telescope_csv = config["validation_telescope_csv"]
    
    # Input and Target Parameters 
    telescope = config["telescope"]
    min_observations = config["min_observations"]
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    input_features = config["input_features"]
    targets = config["targets"]
    target_mode = config["target_mode"]
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    if config["model_constructor"] == 'umonna_unit':
        target_resolutions = get_resolution(targets, target_domains, target_shapes)
    
        # Prepare Generator target_mode_config 
        target_mode_config = {
            "target_shapes":      tuple([target_shapes[target]      for target in targets]),
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_resolutions": tuple([target_resolutions[target] for target in targets])
        }
        if target_mode == "probability_map":
            target_sigmas = config["target_sigmas"]
            target_mode_config["target_sigmas"] = tuple([target_sigmas[target] for target in targets])
    else:
        target_mode_config = {
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_shapes":      tuple([np.inf                     for target in targets]),
            "target_resolutions": tuple([np.inf                     for target in targets])
        }
        target_resolutions = tuple([np.inf      for target in targets])

    # Training Parameters
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    loss = config["loss"]
    optimizer = config["optimizer"]["name"].lower()
    learning_rate = config["optimizer"]["learning_rate"]
    optimizer_parameters = config["optimizer"]["extra_parameters"]
    optimizer_parameters = {} if optimizer_parameters is None else optimizer_parameters
    early_stop_patience = config.get("early_stop_patience", None)

    # Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})

    # Result parameters
    save_checkpoints = config.get("save_checkpoints", True)
    save_predictions = config.get("save_predictions", True)
    save_regressions = config.get("save_regressions", True)
    save_loss = config.get("save_loss", True)

    # Setup Experiment Folder
    experiment_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_run = uuid.uuid4().hex[:3]
    output_folder = path.join(output_folder, f"{model_name}_{experiment_date}_{experiment_run}")
    os.makedirs(output_folder, exist_ok=False)
    print("Experiment Folder:", path.abspath(output_folder))
    experiment_config_file = path.join(output_folder, path.basename(config_file))
    print("Experiment Config file:", experiment_config_file)
    with open(experiment_config_file,  "w") as cfg_file:
        json.dump(config, cfg_file)

    model = train_model(
        # Model parameters
        model_name, model_constructor, assembler_constructor, model_extra_params, telescope,
        # Dataset parameters
        train_events_csv, train_telescope_csv,
        validation_events_csv, validation_telescope_csv, 
        version,
        # Dataset directory parameters
        replace_folder_train, replace_folder_validation,
        output_folder, 
        # Input parameters
        min_observations,
        input_image_mode, input_image_mask, input_features,
        # Target paramters
        target_mode, targets, target_mode_config, 
        # Training paramters
        batch_size = batch_size,
        epochs = epochs, 
        loss = loss, 
        optimizer = optimizer, 
        optimizer_parameters = optimizer_parameters, 
        learning_rate = learning_rate,
        # Preprocessing parameters
        preprocessing_parameters = preprocessing_parameters,
        # Results paramerts
        save_checkpoints = save_checkpoints,
        save_predictions = save_predictions,
        save_regressions = save_regressions,
        save_loss = save_loss,
        # Extra parameters
        quiet=quiet,
        # HPC
        multi_gpu=multi_gpu,
        early_stop_patience=early_stop_patience
    )
