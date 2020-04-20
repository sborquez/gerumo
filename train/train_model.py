import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *

import argparse
import logging
import time
from os import path


import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import plot_model

if __name__ == "__main__":

    """
    argparse
    """
    # Debug
    save_plot = False
    plot_only = False
    summary = False

    # Dataset Parameters
    train_events_csv    = "../dataset/train_events.csv"
    train_telescope_csv = "../dataset/train_telescopes.csv" 
    validation_events_csv    = "../dataset/validation_events.csv"
    validation_telescope_csv = "../dataset/validation_telescopes.csv"
    telescopes = ["MST_FlashCam"]
    min_observations = [3]

    # Data Parameters 
    input_image_mode = "simple-shift"
    input_image_mask = True
    input_features = ["x", "y"]
    target_mode = "probability_map"
    #target_mode = "one_cell"
    #target_mode = "distance"
    targets = ["alt", "az"] #, "log10_mc_energy"]
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
        "target_sigmas":      tuple([target_sigmas[target]      for target in targets]),
        "target_resolutions": tuple([target_resolutions[target] for target in targets])
    }

    # Training Parameters
    batch_size = 32
    epochs = 15
    loss = "crossentropy"
    learning_rate = 1e-1
    save_checkpoints = False
    output_folder = "./output"
    checkpoint_filepath = "%s_%s_e{epoch:03d}_{val_loss:.4f}.h5"%(telescopes[0], loss)
    checkpoint_filepath = path.join(output_folder, checkpoint_filepath)

    # Prepare datasets
    train_dataset      = load_dataset(train_events_csv, train_telescope_csv) 
    validation_dataset = load_dataset(validation_events_csv, validation_telescope_csv)

    train_dataset = aggregate_dataset(train_dataset, az=True, log10_mc_energy=True)
    train_dataset = filter_dataset(train_dataset, telescopes, min_observations, target_domains)
    
    validation_dataset = aggregate_dataset(validation_dataset, az=True, log10_mc_energy=True, hdf5_file=True)
    validation_dataset = filter_dataset(validation_dataset, telescopes, min_observations, target_domains)
    
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
    validation_generator = AssemblerUnitGenerator(validation_dataset, batch_size, 
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
        callbacks.append(
            keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', 
                 verbose=1, save_weights_only=False, mode='min', save_best_only=True))
    
    # Train
    telescope = telescopes[0]
    input_img_shape = INPUT_SHAPE[f"{input_image_mode}-mask" if input_image_mask else input_image_mode][telescope]
    input_features_shape = (len(input_features),)
    target_shapes = target_mode_config["target_shapes"]
    umonna = umonna_unit(telescope, input_image_mode, input_image_mask, 
                    input_img_shape, input_features_shape,
                    targets, target_mode, 
                    target_shapes=target_shapes, 
                    latent_variables=600)
    # Debug
    if summary:
        umonna.summary()
    if save_plot:
        plot_model(umonna, to_file="umonna.png", show_shapes=True)
        plot_model(umonna, to_file="umonna_simple.png", show_shapes=False)
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
    umonna.compile(
        #optimizer=keras.optimizers.Adam(lr=learning_rate),
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.01, nesterov=True),
        loss=loss
    )

    start_time = time.time()
    history = umonna.fit(
        train_generator,
        epochs = epochs,
        verbose = 1,
        validation_data = validation_generator,
        callbacks = callbacks,
        use_multiprocessing = False,
        workers = 5,
        max_queue_size = 20,
    )
    training_time = time.time() - start_time

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








    