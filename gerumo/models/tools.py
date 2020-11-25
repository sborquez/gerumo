"""
Tools
======

Fix, debug or modify models
"""


__all__ = [
    'split_model', 
    'load_model_from_configuration', 'load_model_from_experiment',
    'load_assembler_from_configuration'
]

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def split_model(model, split_layer_name=None, split_layer_index=None, mc_dropout_layer_prefix="bayesian_"):
    """
    Split a trained model for predictions.

    If `split_layer_name` and `split_layer_index` are both provided, `split_layer_index` will take precedence.
    Indices are based on order of horizontal graph traversal (bottom-up).

    Parameters
    ----------
    model : `tensorflow.keras.Model`
        Source trained model.
    split_layer_name : `str` or `None`
        Name of layer to split model. 
    split_layer_index : `int` or `None`
        Telescope type.
    Returns
    =======
        `tuple` of `tensorflow.keras.Model`
        Encoder and Regressor models with source model's weigths.
    """
    # First model
    split_layer_index = split_layer_index or model.layers.index(model.get_layer(split_layer_name))

    encoder = Model(
        model.input,
        model.get_layer(index=split_layer_index).output
    )
    latent_variables_shape = encoder.output.shape[1:]
    # Seconad model
    x = regressor_input = Input(shape=latent_variables_shape)
    for layer in model.layers[split_layer_index + 1:]:
        if mc_dropout_layer_prefix is not None and\
            mc_dropout_layer_prefix in layer.name:
            x = layer(x, training=True)
        else:
            x = layer(x)
    regressor = Model(regressor_input, x)
    ## copy weights
    for layer in regressor.layers[1:]:
        layer.set_weights(
            model.get_layer(name=layer.name).get_weights()
        )
    return encoder, regressor

from os import path
from glob import glob
import numpy as np
import json
from tensorflow.keras.models import load_model

def load_model_from_experiment(experiment_folder, custom_objects, assemblers, epoch=None, get_assembler=True):
    """"
    Load best model using experiment configuration.
    
    Parameters
    ==========
    experiment_folder :  `str`
        Path to experiment folder.
    epoch : `int`, optional
        Selected epoch, if is in None use the last epoch. (default=None)
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load experiment data
    experiment_name = path.basename(experiment_folder)
    checkpoints = glob(path.join(experiment_folder, "checkpoints", "*.h5"))
    checkpoints_by_epochs = {
        int(epoch[-2][1:]) - 1: "_".join(epoch)
        for epoch in map(lambda s: s.split("_"), checkpoints)
    }
    if epoch is None:
        epoch = max(checkpoints_by_epochs.keys())
    elif epoch not in checkpoints_by_epochs:
        epoch = max(filter(lambda e: e < epoch, checkpoints_by_epochs.keys()))

    # Find epoch model
    model_name = f"{experiment_name}_e{epoch}"
    model_or_path = checkpoints_by_epochs[epoch]
    
    # Find configuration file
    config_file = glob(path.join(experiment_folder, "*.json"))
    if len(config_file) != 1:
        raise ValueError("Config file not found in experiment folder", experiment_folder)
    else:
        config_file = config_file[0]
    return load_model_from_configuration(model_or_path, config_file, 
                                         assemblers=assemblers, custom_objects=custom_objects,
                                         model_name=model_name, get_assembler=get_assembler)


def __get_resolution(targets, targets_domain, targets_shape):
    """Return the targets resolution for each target given the targets shape"""
    targets_resolution = {}
    for target in targets:
        vmin, vmax = targets_domain[target]
        shape = targets_shape[target]
        targets_resolution[target]  = (vmax -vmin) / shape
    return targets_resolution


def load_model_from_configuration(model_or_path, config_file, custom_objects, assemblers, model_name=None, get_assembler=True):
    """
    Evaluate model, with configuration file given.
    
    Parameters
    ==========
    model_or_path :  `keras.Model` or `str`
        Loaded keras model or path to hdf5 checkpoint file.
    config_file : `str`
        Path to configuration file
    model_name : `str`, optional
        Replace config model name. (default=None)
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load configuration
    with open(config_file) as cfg_file:
        config = json.load(cfg_file)

    ## Model
    model_name = model_name if model_name is not None else config["model_name"]
    telescope = config["telescope"]
    model = load_model(model_or_path, custom_objects=custom_objects) if isinstance(model_or_path, str) else model_or_path 

    if get_assembler:
        targets = config["targets"]
        target_mode = "lineal"
        target_shapes = config["target_shapes"]
        target_domains = config["target_domains"]
        ## Prepare Generator target_mode_config 
        # TODO: Move this to a function
        if  config["target_shapes"] is not None:
            target_resolutions = __get_resolution(targets, target_domains, target_shapes)
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
                "target_shapes":      tuple([np.inf      for target in targets]),
                "target_resolutions": tuple([np.inf      for target in targets])
            }
            target_resolutions = tuple([np.inf      for target in targets])
        assembler_constructor = assemblers[config["assembler_constructor"]]
        assembler = assembler_constructor(
                targets=targets, 
                target_shapes=target_mode_config["target_shapes"],
                target_domains=target_mode_config["target_domains"],
                target_resolutions=target_mode_config["target_resolutions"],
                point_estimation_mode="expected_value"
        )
        return config, model, assembler
    else:
        return config, model

def load_assembler_from_configuration(assembler_config_file, assemblers):
     # Load configuration
    with open(assembler_config_file) as cfg_file:
        config = json.load(cfg_file)
    ## Model
    model_name = config["model_name"]
    model_name = model_name.replace(' ', '_')
    assembler_constructor = assemblers[config["assembler_constructor"]]
    assembler_mode = config.get("assembler_mode", None)
    telescopes = {t:m for t,m in config["telescopes"].items() if m is not None}

    ## Target Parameters 
    targets = config["targets"]
    target_mode = config["target_mode"]
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    ## Prepare Generator target_mode_config 
    ## TODO: Move this to a function
    if  config["target_shapes"] is not None:
        target_resolutions = __get_resolution(targets, target_domains, target_shapes)
        target_mode_config = {
            "target_shapes":      tuple([target_shapes[target]      for target in targets]),
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_resolutions": tuple([target_resolutions[target] for target in targets])
        }
        if target_mode == "probability_map":
            target_sigmas = config.get("target_sigmas", None)
            target_mode_config["target_sigmas"] = tuple([target_sigmas[target] for target in targets])
    else:
        target_mode_config = {
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_shapes":      tuple([np.inf      for target in targets]),
            "target_resolutions": tuple([np.inf      for target in targets])
        }
        target_resolutions = tuple([np.inf      for target in targets])
    # Assembler
    assembler = assembler_constructor(
            assembler_mode=assembler_mode,
            targets=targets, 
            target_shapes=target_mode_config["target_shapes"],
            target_domains=target_mode_config["target_domains"],
            target_resolutions=target_mode_config["target_resolutions"],
            point_estimation_mode="expected_value"
    )
    # Telescope Models 
    for telescope, experiment_or_model_path in telescopes.items():
        assembler.load_model(telescope, experiment_or_model_path)
    return config, assembler