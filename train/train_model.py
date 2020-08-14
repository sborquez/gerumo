import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *

import logging
import time
from os import path

import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import plot_model


def train_model(model_name, model_constructor, model_extra_params, 
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
            save_checkpoints = True,
            save_plot=False, plot_only=False, summary=False, quiet=False):

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
        checkpoint_filepath = "%s_%s_%s_e{epoch:03d}_{val_loss:.4f}.h5"%(model_name, telescope, loss)
        checkpoint_filepath = path.join(output_folder, checkpoint_filepath)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', 
                 verbose=1, save_weights_only=False, mode='min', save_best_only=True))
    
    # Train
    input_img_shape = INPUT_SHAPE[f"{input_image_mode}-mask" if input_image_mask else input_image_mode][telescope]
    input_features_shape = (len(input_features),)
    target_shapes = target_mode_config["target_shapes"]
    model = model_constructor(telescope, input_image_mode, input_image_mask, 
                    input_img_shape, input_features_shape,
                    targets, target_mode, target_shapes, 
                    **model_extra_params)
    # Debug
    if summary:
        model.summary()
    if save_plot:
        # TODO: Move this to debug
        plot_model(model, to_file="model.png", show_shapes=True)
        plot_model(model, to_file="model_simple.png", show_shapes=False)
        if plot_only:
            exit(0)

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

    ## fit
    model.compile(
        optimizer=optimizer_,
        loss=loss_
    )

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
    args = vars(ap.parse_args()) 
    config_file = args["config"]
    quiet = args["quiet"]
    
    print(f"Loading config from: {config_file}")
    with open(config_file) as cfg_file:
        config = json.load(cfg_file)

    # Model
    model_name = config["model_name"]
    model_constructor = MODELS[config["model_constructor"]]
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
    if config["model_constructor"] == 'umonna':
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

    # Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})

    # Result parameters
    save_checkpoints = config["save_checkpoints"]
    save_plot = config["save_plot"]
    
    # Debug
    plot_only = config["plot_only"]
    summary = config["summary"]

    model = train_model(
        # Model parameters
        model_name, model_constructor, model_extra_params, telescope,
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
        batch_size, epochs, 
        loss, 
        optimizer, optimizer_parameters, learning_rate,
        # Preprocessing parameters
        preprocessing_parameters,
        # Results paramerts
        save_checkpoints, save_plot, 
        # Debug parameters
        plot_only, summary, quiet 
    )
