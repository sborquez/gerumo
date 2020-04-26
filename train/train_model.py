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


def get_default_target_mode_config(targets=["alt", "az", "log10_mc_energy"], return_sigmas=True):
    target_shapes = {
        'alt': 81, 
        'az': 81, 
        'log10_mc_energy': 81
    }
    target_domains = {
        'alt': (1.05, 1.382), 
        'az': (-0.52, 0.52), 
        'log10_mc_energy': (-1.74, 2.44)
    }
    target_sigmas = {
        'alt': 0.002, 
        'az':  0.002, 
        'log10_mc_energy': 0.002
    }
    target_resolutions = get_resolution(targets, target_domains, target_shapes)
    # Prepare Generator target_mode_config 
    target_mode_config = {
        "target_shapes":      tuple([target_shapes[target]      for target in targets]),
        "target_domains":     tuple([target_domains[target]     for target in targets]),
        "target_resolutions": tuple([target_resolutions[target] for target in targets])
    }
    if return_sigmas:
        target_mode_config["target_sigmas"] = tuple([target_sigmas[target] for target in targets])
    return target_mode_config

def load_config(json_path):
    pass

def train_model(model_name, model_constructor, model_extra_params, 
            telescope, 
            train_events_csv, train_telescope_csv,
            validation_events_csv, validation_telescope_csv,
            replace_folder_train = None,
            replace_folder_validation = None,
            output_folder = "./output", 
            min_observations = 3,
            input_image_mode = "simple-shift",
            input_image_mask = True,
            input_features = ["x", "y"],
            target_mode = "probability_map",
            targets = ["alt", "az", "log10_mc_energy"],
            target_mode_config = get_default_target_mode_config(),
            batch_size = 32,
            epochs = 3,
            loss = "crossentropy",
            learning_rate = 1e-1,
            save_checkpoints = True,
            save_plot=False, plot_only=False, summary=False):

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
    preprocess_input_pipes = []
    preprocess_output_pipes = []

    # Generators
    train_generator = AssemblerUnitGenerator(train_dataset, batch_size, 
                                            input_image_mode=input_image_mode, 
                                            input_image_mask=input_image_mask, 
                                            input_features=input_features,
                                            targets=targets,
                                            target_mode=target_mode, 
                                            target_mode_config=target_mode_config,
                                            preprocess_input_pipes=preprocess_input_pipes,
                                            preprocess_output_pipes=preprocess_output_pipes
                                            )
    validation_generator = AssemblerUnitGenerator(validation_dataset, batch_size//4, 
                                                    input_image_mode=input_image_mode,
                                                    input_image_mask=input_image_mask, 
                                                    input_features=input_features,
                                                    targets=targets,
                                                    target_mode=target_mode, 
                                                    target_mode_config=target_mode_config,
                                                    preprocess_input_pipes=preprocess_input_pipes,
                                                    preprocess_output_pipes=preprocess_output_pipes
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
                    targets, target_mode, 
                    target_shapes=target_shapes, 
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
    if loss == "crossentropy":
        loss = crossentropy_loss(dimensions=len(targets))
    elif loss == "hellinger":
        loss = hellinger_loss()
    elif loss == "mse":
        loss = mse_loss()
    elif loss == "distance":
        loss = mean_distance_loss(target_shapes)
    
    ## fit
    model.compile(
        #optimizer=keras.optimizers.Adam(lr=learning_rate),
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.01, nesterov=True),
        loss=loss
    )

    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs = epochs,
        verbose = 1,
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

    # # Validate
    # i = 6
    # batch_0 = train_generator[120]
    # prediction = umonna.predict(batch_0[0])[i]
    # target = batch_0[1][i]

    # import matplotlib.pyplot as plt

    # plt.imshow(target)
    # plt.title("Target_2")
    # plt.show()

    # plt.imshow(prediction)
    # plt.title("Prediction_2")
    # plt.show()

    return model

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Train a model.")
    ap.add_argument("-c", "--config", type=str, default=None, help="Configuration file for model/experiment.")
    args = vars(ap.parse_args()) 
    config_file = args["config"]
    if config_file is None:
        print("Loading default configuration.")
        # Model
        model_name = "UMONNA_UNIT_MST_V2"
        model_constructor = umonna_unit
        model_extra_params = {"latent_variables": 600}

        # Dataset Parameters
        output_folder = "./output"
        replace_folder_train = "D:/sebas/Google Drive/Projects/gerumo/dataset"
        replace_folder_validation = "D:/sebas/Google Drive/Projects/gerumo/dataset"
        train_events_csv    = "../dataset/train_events.csv"
        train_telescope_csv = "../dataset/train_telescopes.csv" 
        validation_events_csv    = "../dataset/validation_events.csv"
        validation_telescope_csv = "../dataset/validation_telescopes.csv"
        
        # Input and Target Parameters 
        telescope = "MST_FlashCam"
        min_observations = 3
        input_image_mode = "simple-shift"
        input_image_mask = True
        input_features = ["x", "y"]
        targets = ["alt", "az"] #,"log10_mc_energy"]
        target_mode = "probability_map"
        if target_mode == "probability_map":
            target_mode_config = get_default_target_mode_config(targets, True)
        else:
            target_mode_config = get_default_target_mode_config(targets, False)

        # Training Parameters
        batch_size = 32
        epochs = 3
        loss = "crossentropy"
        learning_rate = 1e-1
        save_checkpoints = True

        # Debug
        save_plot = False
        plot_only = False
        summary = False
    else:
        print(f"Loading config from: {config_file}")
        with open(config_file) as cfg_file:
            config = json.load(cfg_file)

        # Model
        model_name = config["model_name"]
        model_constructor = MODELS[config["model_constructor"]]
        model_extra_params = config["model_extra_params"]

        # Dataset Parameters
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

        # Training Parameters
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        loss = config["loss"]
        learning_rate = config["learning_rate"]
        save_checkpoints = config["save_checkpoints"]

        # Debug
        save_plot = config["save_plot"]
        plot_only = config["plot_only"]
        summary = config["summary"]

    model = train_model(
        model_name, model_constructor, model_extra_params, telescope,
        train_events_csv, train_telescope_csv,
        validation_events_csv, validation_telescope_csv,
        replace_folder_train, replace_folder_validation,
        output_folder, min_observations,
        input_image_mode, input_image_mask, input_features,
        target_mode, targets, target_mode_config, 
        batch_size, epochs, loss, learning_rate, save_checkpoints, 
        save_plot, plot_only, summary 
    )   