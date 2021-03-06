import sys
sys.path.insert(1, '..')

from gerumo import *


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import timeline

from time import time
import uuid
from os.path import abspath, basename, splitext, isdir, join
from glob import glob
from os import remove, makedirs
import argparse

import logging


def test_gpu():
    cuda = tf.test.is_built_with_cuda()
    if cuda:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            #logging.info(f"Availables GPUs: {len(gpus)}" )
            print(f"Availables GPUs: {len(gpus)}" )
            for i, gpu in enumerate(gpus):
                #logging.info(f"Availables GPU {i}: {gpu}" )
                print(f"Availables GPU {i}: {gpu}" )
        else:
            #logging.info("Not availables GPUs.")
            print("Not availables GPUs.")
    else:
        #logging.info("Tensorflow is not built with CUDA")
        print("Tensorflow is not built with CUDA")

def test_saving(model="umonna"):
    if model == "umonna":
        # SETUP CONFIG
        telescope = "MST_FlashCam"
        input_image_mode = "simple-shift"
        input_image_mask = True
        input_features = ["x", "y"]
        target_mode = "probability_map"
        targets = ["alt", "az", "log10_mc_energy"]
        target_shapes = { 'alt': 81, 'az': 81, 'log10_mc_energy': 81 }
        target_domains = {'alt': (1.05, 1.382), 'az': (-0.52, 0.52), 'log10_mc_energy': (-1.74, 2.44)}
        target_sigmas = {'alt': 0.002, 'az':  0.002, 'log10_mc_energy': 0.002}
        target_resolutions = get_resolution(targets, target_domains, target_shapes)
        target_mode_config = {
            "target_shapes":      tuple([target_shapes[target]      for target in targets]),
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_sigmas":      tuple([target_sigmas[target]      for target in targets]),
            "target_resolutions": tuple([target_resolutions[target] for target in targets])
        }
        target_shapes = target_mode_config["target_shapes"]
        input_img_shape = INPUT_SHAPE[f"{input_image_mode}-mask" if input_image_mask else input_image_mode][telescope]
        input_features_shape = (len(input_features),)
        umonna = umonna_unit(telescope, input_image_mode, input_image_mask, input_img_shape, input_features_shape,
                        targets, target_mode, target_shapes=target_shapes, latent_variables=600)
        loss = crossentropy_loss(dimensions=len(targets))
        umonna.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.01, nesterov=True),
            loss=loss
        )
        try:
            umonna.save("test_saving.h5")
            u1_layer_weigths = umonna.layers[5].weights[0].numpy()
            del umonna
            CUSTOM_OBJECTS["loss"] = loss
            umonna = load_model("test_saving.h5", CUSTOM_OBJECTS)
            u2_layer_weigths = umonna.layers[5].weights[0].numpy()
            are_equals = (u1_layer_weigths == u2_layer_weigths).all()
            if are_equals:
                print("Saving Correctly")
            else:
                print("Weights are not equals.")
            #remove("test_saving.h5")
        except Exception as err:
            print(err)
            print("Unable to save/load model.")


def bottle_neck(model_path):
    raise NotImplementedError

def save_plot_model(experiment_or_model_path, output_folder, include_shape=False):

    # load model
    if isdir(experiment_or_model_path):
        model_path = glob(join(experiment_or_model_path, 'checkpoints' , "*.h5"))[0]
    elif splitext(experiment_or_model_path)[1] == ".h5":
        model_path = experiment_or_model_path
    else:
        raise ValueError("Invalid experiment_or_model_path", experiment_or_model_path)
    makedirs(output_folder, exist_ok=True)

    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS) if isinstance(model_path, str) else model_path

    # save plot
    model_image = join(output_folder, "model.png")
    print("Saving model in", )
    if include_shape:
        plot_model(model, to_file=model_image, show_shapes=True)
    else:
        plot_model(model, to_file=model_image, show_shapes=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Profiling and debuging tools to find bottlenecks in the architecture and other tests.")
    # option a
    ap.add_argument("--gpu", dest="gpu", action="store_true", help="Check GPU availability.")
    # option b
    ap.add_argument("--save", dest="save", action="store_true", help="Check if model can be saved")
    # option c
    ap.add_argument("--plot", type=str, default=None, help="Path to model checkpoint or experiment path.")
    ap.add_argument("--include_shape", action="store_true", help="Include shape in model plot.")
    ap.add_argument("-o", "--output", type=str, default=".", help="Output folder path")
    args = vars(ap.parse_args())
    
    gpu = args["gpu"]
    if gpu:
        print("Running GPU tests.")
        test_gpu()
    
    save = args["save"]
    if save:
        test_saving()
    
    model = args["plot"]
    if model is not None:
        print("Saving model plot.")
        output_folder = args["output"]
        include_shape = args["include_shape"]
        save_plot_model(model, output_folder, include_shape=include_shape)

    print("Done")