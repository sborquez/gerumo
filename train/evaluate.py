import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *
import json
import logging

import time
import os
from os import path
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model, Model
import tensorflow_probability as tfp
import matplotlib as mpl
mpl.use('Agg')


def same_telescopes(src_telescopes, sample_telescopes):
    return set(sample_telescopes).issubset(set(src_telescopes))
    
# Evaluate
# 1. Save results: points, predictions (optional), targets, energy, event info and telescope info
# 2. Calculate regression
# 3. Calculate resolution (angular or energy)
# 4. Prediction Samples (optinal)

def evaluate_experiment_folder(experiment_folder,  save_results=True, save_predictions=True, save_samples=True, epoch=None, replace_folder_test=None, seed=None):
    """"
    Evaluate best model using experiment configuration.
    
    Parameters
    ==========
    experiment_folder :  `str`
        Path to experiment folder.
    save_results : `bool`, optional
        Save predictions points and targets into csv file.
    save_predictions : `bool`, optional
        Save raw predictions.
    save_samples : `bool`, optional
        Save predictions plots from a small events sampleset .
    epoch : `int`, optional
        Selected epoch, if is in None use the last epoch. (default=None)
    replace_folder_test : `str`, optional
        Replace config `replace_folder_test` without edit configuration file.
    seed : `int`, optional
        Seed for random states.
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load experiment data
    experiment_name = path.basename(experiment_folder)
    checkpoints = glob(path.join(experiment_folder, "checkpoints", "*.h5"))
    checkpoints_by_epochs = {int(epoch[-2][1:]) - 1: "_".join(epoch) for epoch in map(lambda s: s.split("_"), checkpoints)}
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
    
    # Evaluate
    output_folder = path.join(experiment_folder, 'evaluation')
    os.makedirs(output_folder, exist_ok=True)
    print("Making evaluation folder:", output_folder)
    return evaluate_unit(model_or_path, config_file, output_folder,
                  save_predictions=save_predictions, save_samples=save_samples,
                  model_name=model_name, replace_folder_test=replace_folder_test, seed=seed)

def evaluate_unit(model_or_path, config_file, output_folder,
                 assembler=None, telescope=None,
                 save_results=True, save_predictions=True, save_samples=True, 
                 model_name=None, replace_folder_test=None, seed=None, sample_telescopes=None):
    """
    Evaluate model, with configuration file given.
    
    Parameters
    ==========
    model_or_path :  `keras.Model` or `str`
        Loaded keras model or path to hdf5 checkpoint file.
    config_file : `str`
        Path to configuration file
    output_folder : `str`
        Path to folder where plots and results will be saved.
    telescope : `str`, optional
        Model's telescope type.
    assembler : `gerumo.Assembler`, optional
        Assembler model instance.
    save_results : `bool`, optional
        Save predictions points and targets into csv file.
    save_predictions : `bool`, optional
        Save raw predictions.
    save_samples : `bool`, optional
        Save predictions plots from a small events sampleset .
    model_name : `str`, optional
        Replace config model name. (default=None)
    replace_folder_test : `str`, optional
        Replace config `replace_folder_test` without edit configuration file.
    seed : `int`, optional
        Seed for random states.
    sample_telescopes : `list` of `str` or `None`
        Filter sample dataset.
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
    assembler_constructor = ASSEMBLERS[config["assembler_constructor"]]
    telescope = config["telescope"] if telescope is None else telescope
    model = load_model(model_or_path, custom_objects=CUSTOM_OBJECTS) if isinstance(model_or_path, str) else model_or_path 

    # Prepare datasets
    version = config["version"]
    test_events_csv    = config["test_events_csv"] 
    test_telescope_csv = config["test_telescope_csv"]
    replace_folder_test_ = config["replace_folder_test"] 
    if (test_events_csv is None) or (test_telescope_csv is None):
        print("Using validation dataset")
        test_events_csv    = config["validation_events_csv"]
        test_telescope_csv = config["validation_telescope_csv"]
        replace_folder_test_ = config["replace_folder_validation"]
    replace_folder_test = replace_folder_test_ if replace_folder_test is None else replace_folder_test

    ## Input Parameters 
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    min_observations = config["min_observations"]
    input_features = config["input_features"]
    
    ## Target Parameters 
    targets = config["targets"]
    target_mode = "lineal"
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    ## Prepare Generator target_mode_config 
    if  config["target_shapes"] is not None:
        target_resolutions = get_resolution(targets, target_domains, target_shapes)
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

    ## Load Data
    test_dataset = load_dataset(test_events_csv, test_telescope_csv, replace_folder_test)
    test_dataset = aggregate_dataset(test_dataset, az=True, log10_mc_energy=True)
    print("Test dataset")
    describe_dataset(test_dataset, save_to=path.join(output_folder, "test_description.txt"))
    if save_samples:
        # events with observations of every type of telescopes
        sample_telescopes = sample_telescopes if sample_telescopes is not None else [telescope]
        sample_events = [e for e, df in test_dataset.groupby("event_unique_id") if same_telescopes(df["type"].unique(), sample_telescopes)]
        # TODO: add custom seed
        r = np.random.RandomState(42)
        sample_events = r.choice(sample_events, size=5, replace=False)
        sample_dataset = test_dataset[test_dataset["event_unique_id"].isin(sample_events)]
        sample_dataset = filter_dataset(sample_dataset, telescope, [0], target_domains)

        print("Sample dataset")
        describe_dataset(sample_dataset, save_to=path.join(output_folder, "sample_description.txt"))
    else:
        sample_dataset = None
        sample_generator = None
    test_dataset = filter_dataset(test_dataset, telescope, [0], target_domains)
    
    ## Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})

    # Preprocessing pipes
    ## input preprocessing
    preprocess_input_pipes = {}
    if "CameraPipe" in preprocessing_parameters:
        camera_parameters = preprocessing_parameters["CameraPipe"]
        camera_pipe = CameraPipe(telescope_type=telescope, version=version, **camera_parameters)
        preprocess_input_pipes['CameraPipe'] = camera_pipe
    elif "MultiCameraPipe" in preprocessing_parameters:
        camera_parameters = preprocessing_parameters["MultiCameraPipe"][telescope]
        camera_pipe = CameraPipe(telescope_type=telescope, version=version, **camera_parameters)
        preprocess_input_pipes['CameraPipe'] = camera_pipe

    if "TelescopeFeaturesPipe" in preprocessing_parameters:
        telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
        telescope_features_pipe = TelescopeFeaturesPipe(version=version, **telescopefeatures_parameters)
        preprocess_input_pipes['TelescopeFeaturesPipe'] = telescope_features_pipe
    ## output preprocessing
    preprocess_output_pipes = {}

    ## Dataset Generators
    test_generator =  AssemblerUnitGenerator(
                            test_dataset, 16, 
                            input_image_mode=input_image_mode,
                            input_image_mask=input_image_mask, 
                            input_features=input_features,
                            targets=targets,
                            target_mode="lineal", 
                            target_mode_config=target_mode_config,
                            preprocess_input_pipes=preprocess_input_pipes,
                            preprocess_output_pipes=preprocess_output_pipes,
                            include_event_id=True,
                            include_true_energy=True,
                            version=version
                        )
    if save_samples:
        sample_generator =  AssemblerUnitGenerator(
                sample_dataset, min(16, len(sample_dataset)), 
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
    else:
        sample_generator = None

    # Assembler
    if assembler is None:
        assembler = assembler_constructor(
                targets=targets, 
                target_shapes=target_mode_config["target_shapes"],
                target_domains=target_mode_config["target_domains"],
                target_resolutions=target_mode_config["target_resolutions"],
                point_estimation_mode="expected_value"
        )

    # Evaluate
    # 0. Evaluate with generator
    results, predictions = assembler.exec_model_evaluate(model, telescope, test_generator, return_predictions=True)

    # 1. Save results in csv file: points, targets, energy, event info and telescope info
    if save_results:
        results.to_csv(path.join(output_folder, "results.csv"), index=False)
    if save_predictions:
        # Create predictions folder
        predictions_subfolder = path.join(output_folder, 'predictions')
        print("Saving predictions in:", predictions_subfolder)
        os.makedirs(predictions_subfolder, exist_ok=True)
        for _, row in results.iterrows():
            prediction_i = row["predictions"]
            prediction = predictions[prediction_i]
            prediction_filepath = path.join(predictions_subfolder, f"{prediction_i}.npy")
            if isinstance(prediction, np.ndarray):
                np.save(prediction_filepath, prediction)
            elif isinstance(prediction, tfp.distributions.MultivariateNormalTriL) :
                mu = prediction.mean().numpy()
                cov = prediction.covariance().numpy()
                np.save(prediction_filepath, np.vstack((mu, cov)))
            elif isinstance(prediction, st.gaussian_kde):
                np.save(prediction_filepath, prediction.dataset)
            else:
                raise NotImplemented("Unknown prediction type:", type(prediction))

    
    # 2. Save samples plots
    if save_samples:
        # makedir
        samples_subfolder = path.join(output_folder, 'samples')
        print("Saving samples in:", samples_subfolder)
        os.makedirs(samples_subfolder, exist_ok=True)
        results_samples, inputs_samples, predictions_samples = assembler.exec_model_evaluate(model, telescope, sample_generator, return_inputs=True, return_predictions=True)

        for _, row in results_samples.iterrows():
            # event  and telescope info
            event_id = row["event_id"]
            telescope_id =  row["telescope_id"]
            # input sample
            input_sample = inputs_samples[row["inputs_values"]]
            input_image_sample, input_features_sample = input_sample
            prediction_sample = predictions_samples[row["predictions"]]
            # prediction sample
            # prediction point sample
            pred_targets = [f"pred_{target}" for target in targets]
            prediction_sample_point = row[pred_targets].to_numpy()
            # Target sample
            true_targets = [f"true_{target}" for target in targets]
            target_sample = row[true_targets].to_numpy() 
            # Plot input
            image_filepath = path.join(samples_subfolder, f"event_id_{event_id}_telescope_id_{telescope_id}_input.png")
            plot_input_sample(input_image_sample, input_image_mode, input_features_sample, title=(event_id, telescope_id), make_simple=True,
                                save_to=image_filepath)
            # Plot prediction
            prediction_filepath = path.join(samples_subfolder, f"event_id_{event_id}_telescope_id_{telescope_id}_prediction.png")
            plot_prediction(prediction_sample, prediction_sample_point, targets,
                            target_domains, target_resolutions, 
                            (event_id, telescope_id), target_sample, save_to=prediction_filepath)

    ## 3. Calculate regression
    print("Regression plots")
    scores = r2_score(results[[f"true_{target}" for target in targets]], results[[f"pred_{target}" for target in targets]], multioutput="raw_values")
    plot_regression_evaluation(results, targets, scores, save_to=path.join(output_folder, "regression.png"))

    reconstruction_mode = assembler.select_mode(targets)
    if reconstruction_mode in ("angular reconstruction", "complete reconstruction"):
        print("Angular Reconstruction")
        ## 3. Calculate resolution (angular)
        plot_error_and_angular_resolution(results, save_to=path.join(output_folder, "angular_resolution.png"))
        plot_angular_resolution_comparison({telescope: results}, ylim=[0, 2], save_to=path.join(output_folder, "angular_resolution_comparable.png"))

    if reconstruction_mode in ("energy reconstruction", "complete reconstruction"):
        print("Energy resolution")
        ## 3. Calculate resolution (energy)
        plot_error_and_energy_resolution(results, save_to=path.join(output_folder, "energy_resolution.png"))
        plot_energy_resolution_comparison({telescope: results}, ylim=[0, 2], save_to=path.join(output_folder, "energy_resolution_comparable.png"))

    if save_predictions:
        return results, predictions

    return results


def evaluate_assembler(assembler_config_file, output_folder=None, save_all_unit_evaluations=True, save_results=True, save_predictions=True, save_samples=True, seed=None):
    """
    Evaluate assembler from a configuration file.
    
    Parameters
    ==========
    assembler_config_file :  `str`
        Path to configuration file.
    output_folder : `str`, optional
        Path to folder where plots and results will be saved (default: use the output from configuration file) 
    save_all_unit_evaluations : `bool`
        Evaluate each telescope.
    save_results : `bool`, optional
        Save predictions points and targets into csv file.
    save_predictions : `bool`, optional
        Save raw predictions.
    save_samples : `bool`, optional
        Save predictions plots from a small events sampleset .
    seed : `int`, optional
        Seed for random states.
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load configuration
    with open(assembler_config_file) as cfg_file:
        config = json.load(cfg_file)

    ## Model
    model_name = config["model_name"]
    model_name = model_name.replace(' ', '_')
    assembler_constructor = ASSEMBLERS[config["assembler_constructor"]]
    assembler_mode = config.get("assembler_mode", None)
    telescopes = {t:m for t,m in config["telescopes"].items() if m is not None}
    output_folder = config["output_folder"] if output_folder is None else output_folder
    output_folder = path.join(output_folder, f"{model_name}_evaluation")
    print("Saving assembelr evaluation in:", output_folder)
    os.makedirs(output_folder, exist_ok=True)
    with open(path.join(output_folder, f"{model_name}.json"),  "w") as cfg_file:
        json.dump(config, cfg_file)
    
    # Prepare datasets
    version = config["version"]
    test_events_csv    = config["test_events_csv"] 
    test_telescope_csv = config["test_telescope_csv"]
    replace_folder_test = config["replace_folder_test"]

    ## Input Parameters 
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    min_observations = config["min_observations"]
    input_features = config["input_features"]
    
    ## Target Parameters 
    targets = config["targets"]
    target_mode = "lineal"
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    ## Prepare Generator target_mode_config 
    if  config["target_shapes"] is not None:
        target_resolutions = get_resolution(targets, target_domains, target_shapes)
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

    ## Load Data
    test_dataset = load_dataset(test_events_csv, test_telescope_csv, replace_folder_test)
    test_dataset = aggregate_dataset(test_dataset, az=True, log10_mc_energy=True)
    print("Test dataset")
    describe_dataset(test_dataset, save_to=path.join(output_folder, "test_description.txt"))
    if save_samples:
        # events with observations of every type of telescopes
        sample_telescopes = [t for t,p in telescopes.items() if p is not None]
        sample_events = [e for e, df in test_dataset.groupby("event_unique_id") if same_telescopes(df["type"].unique(), sample_telescopes)]
        # TODO: add custom seed
        r = np.random.RandomState(42)
        sample_events = r.choice(sample_events, size=5, replace=False)
        sample_dataset = test_dataset[test_dataset["event_unique_id"].isin(sample_events)]
        sample_dataset = filter_dataset(sample_dataset, telescopes.keys(), min_observations, target_domains)

        print("Sample dataset")
        describe_dataset(sample_dataset, save_to=path.join(output_folder, "sample_description.txt"))
        if len(sample_dataset) == 0:
            raise ValueError("Sample dataset is empty.")
    else:
        sample_telescopes = None
        sample_dataset = None
        sample_generator = None
    test_dataset = filter_dataset(test_dataset, telescopes.keys(), min_observations, target_domains)
    
    # Assembler
    assembler = assembler_constructor(
            assembler_mode=assembler_mode,
            targets=targets, 
            target_shapes=target_mode_config["target_shapes"],
            target_domains=target_mode_config["target_domains"],
            target_resolutions=target_mode_config["target_resolutions"],
            point_estimation_mode="expected_value"
    )

    if save_all_unit_evaluations:
        all_results = {}

    # Telescope Models 
    for telescope, experiment_or_model_path in telescopes.items():
        model_path = assembler.load_model(telescope, experiment_or_model_path)
        # (Optional) Evaluate each telescope model
        if save_all_unit_evaluations:
            model_output_folder = path.join(output_folder, 'telescopes', telescope)
            print(f"Saving {telescope} evaluation in:", model_output_folder)
            os.makedirs(model_output_folder, exist_ok=True)
            unit_evaluation = evaluate_unit(model_path, assembler_config_file, model_output_folder,
                 assembler=assembler, telescope=telescope,
                 save_results=save_results, save_predictions=save_predictions, save_samples=save_samples, 
                 model_name=model_name, seed=seed, sample_telescopes=sample_telescopes)
            if save_predictions:
                all_results[telescope] = unit_evaluation[0]
            else:    
                all_results[telescope] = unit_evaluation

    # Evaluate assembler
    ## Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})

    # Preprocessing pipes
    ## input preprocessing
    preprocess_input_pipes = {}
    if ("MultiCameraPipe" in preprocessing_parameters):
        multicamera_parameters = preprocessing_parameters["MultiCameraPipe"]
        multicamera_pipe = MultiCameraPipe(version=version, **multicamera_parameters)
        preprocess_input_pipes['MultiCameraPipe'] = multicamera_pipe
    if "TelescopeFeaturesPipe" in preprocessing_parameters:
        telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
        telescope_features_pipe = TelescopeFeaturesPipe(version=version, **telescopefeatures_parameters)
        preprocess_input_pipes['TelescopeFeaturesPipe'] = telescope_features_pipe
    ## output preprocessing
    preprocess_output_pipes = {}

    ## Dataset Generators
    test_generator =  AssemblerGenerator(
                            test_dataset, telescopes.keys(), 16, 
                            input_image_mode=input_image_mode,
                            input_image_mask=input_image_mask, 
                            input_features=input_features,
                            targets=targets,
                            target_mode="lineal", 
                            target_mode_config=target_mode_config,
                            preprocess_input_pipes=preprocess_input_pipes,
                            preprocess_output_pipes=preprocess_output_pipes,
                            include_event_id=True,
                            include_true_energy=True,
                            version=version
                        )
    if save_samples:
        sample_generator =  AssemblerGenerator(
                sample_dataset, telescopes.keys(), min(16, len(sample_dataset)), 
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
    else:
        sample_generator = None
    
    results, predictions = assembler.evaluate(test_generator, return_predictions=True)
    
    # 1. Save results in csv file: points, targets, energy, event info and telescope info
    if save_results:
        results.to_csv(path.join(output_folder, "results.csv"), index=False)
    if save_predictions:
        # Create predictions folder
        predictions_subfolder = path.join(output_folder, 'predictions')
        print("Saving predictions in:", predictions_subfolder)
        os.makedirs(predictions_subfolder, exist_ok=True)
        for _, row in results.iterrows():
            prediction_i = row["predictions"]
            prediction = predictions[prediction_i]
            prediction_filepath = path.join(predictions_subfolder, f"{prediction_i}.npy")
            if isinstance(prediction, np.ndarray):
                np.save(prediction_filepath, prediction)
            elif isinstance(prediction, tfp.distributions.MultivariateNormalTriL) :
                mu = prediction.mean().numpy()
                cov = prediction.covariance().numpy()
                np.save(prediction_filepath, np.vstack((mu, cov)))
            elif isinstance(prediction, st.gaussian_kde):
                np.save(prediction_filepath, prediction.dataset)
            else:
                raise NotImplemented("Unknown prediction type:", type(prediction))

    # 2. Save samples plots
    if save_samples:
        # makedir
        samples_subfolder = path.join(output_folder, 'samples')
        print("Saving samples in:", samples_subfolder)
        os.makedirs(samples_subfolder, exist_ok=True)
        results_samples, predictions_samples = assembler.evaluate(sample_generator, return_predictions=True)
        for _, row in results_samples.iterrows():
            # event  and telescope info
            event_id = row["event_id"]
            # prediction sample
            prediction_sample = predictions_samples[row["predictions"]]
            # prediction point sample
            pred_targets = [f"pred_{target}" for target in targets]
            prediction_sample_point = row[pred_targets].to_numpy()
            # Target sample
            true_targets = [f"true_{target}" for target in targets]
            target_sample = row[true_targets].to_numpy() 

            # Plot prediction
            prediction_filepath = path.join(samples_subfolder, f"event_id_{event_id}_prediction.png")
            plot_prediction(prediction_sample, prediction_sample_point, targets,
                            target_domains, target_resolutions, 
                            event_id, target_sample, save_to=prediction_filepath)

    ## 3. Calculate regression
    print("Regression plots")
    scores = r2_score(results[[f"true_{target}" for target in targets]], results[[f"pred_{target}" for target in targets]], multioutput="raw_values")
    plot_regression_evaluation(results, targets, scores, save_to=path.join(output_folder, "regression.png"))

    reconstruction_mode = assembler.select_mode(targets)
    if reconstruction_mode in ("angular reconstruction", "complete reconstruction"):
        print("Angular Reconstruction")
        ## 3. Calculate resolution (angular)
        plot_error_and_angular_resolution(results, save_to=path.join(output_folder, "angular_resolution.png"))
        plot_angular_resolution_comparison({model_name: results}, ylim=[0, 2], save_to=path.join(output_folder, "angular_resolution_comparable.png"))
        if save_all_unit_evaluations:
            all_results[model_name] = results
            plot_angular_resolution_comparison(all_results, ylim=[0, 2], save_to=path.join(output_folder, "angular_resolution_comparison.png"))

    if reconstruction_mode in ("energy reconstruction", "complete reconstruction"):
        print("Energy resolution")
        ## 3. Calculate resolution (energy)
        plot_error_and_energy_resolution(results, save_to=path.join(output_folder, "energy_resolution.png"))
        plot_energy_resolution_comparison({model_name: results}, ylim=[0, 2], save_to=path.join(output_folder, "energy_resolution_comparable.png"))
        if save_all_unit_evaluations:
            all_results[model_name] = results
            plot_energy_resolution_comparison(all_results, ylim=[0, 2], save_to=path.join(output_folder, "energy_resolution_comparison.png"))

    if save_predictions:
        return results, predictions
    return results
    
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate models.")
    # Option 1
    ap.add_argument("-a", "--assembler", type=str, default=None, 
                    help="Assembler configuration file.")
    ap.add_argument("--all", action="store_true", dest="save_all_unit_evaluations",
                     help="Evaluate each telescope.")
    # Option 2
    ap.add_argument("-e", "--experiment", type=str, default=None, 
                    help="Experiment folder.")
    ap.add_argument("--epoch", type=int, default=None, 
                    help="Select epoch from experiment folder.")
    # Option 3
    ap.add_argument("-m", "--model", type=str, default=None, 
                    help="Model checkpoint. (require configuration file)")
    ap.add_argument("-c", "--config", type=str, default=None, 
                    help="Model configuration file. (require model checkpoint)")
    ## Plots
    ap.add_argument("--results", action="store_true", dest="save_results",
                     help="Save results into csv file.")
    ap.add_argument("--samples", action="store_true", dest="save_samples",
                     help="Save inputs and predictions plots from sample dataset.")
    ap.add_argument("--predictions", action="store_true", dest="save_predictions",
                     help="Save predictions from test dataset.")
    ## Evaluation options
    ap.add_argument("--output", type=str, default=None,
                    help="Output folder.")
    ap.add_argument("--seed", type=int, default=None, 
                    help="Set random state seed. (not implemented)") # TODO: implement seed 
    ap.add_argument("--replace_folder_test", type=str, default=None,
                    help="Replace test folder without edit configuration files.")
    args = vars(ap.parse_args()) 

    # Model/Assembler parameters
    experiment_folder = args["experiment"] 
    assembler_config = args["assembler"]
    model_checkpoint = args["model"]
    model_config = args["config"]

    # Evaluation parameters
    replace_folder_test = args["replace_folder_test"]
    save_predictions = args["save_predictions"]
    save_samples = args["save_samples"]
    save_results = args["save_results"]
    seed = args["seed"]
    
    if assembler_config is not None:
        output_folder = args["output"]
        evaluate_assembler(assembler_config, 
                          output_folder=output_folder,
                          save_all_unit_evaluations=args["save_all_unit_evaluations"], 
                          save_results=save_results, save_predictions=save_predictions, save_samples=save_samples, 
                          seed=seed)
    elif experiment_folder is not None:
        epoch = args["epoch"]
        evaluate_experiment_folder( experiment_folder, 
                                    save_results=save_results, save_predictions=save_predictions, save_samples=save_samples,
                                    epoch=epoch, replace_folder_test=replace_folder_test, seed=seed)
    elif (model_config is not None) and (model_checkpoint is not None):
        output_folder = "." if args["output"] is None else args["output"]
        output_folder = path.join(output_folder, 'evaluation')
        os.makedirs(output_folder, exist_ok=True)
        print("Making evaluation folder:", output_folder)
        evaluate_unit( model_checkpoint, model_config, output_folder,
                       save_results=save_results, save_predictions=save_predictions, save_samples=save_samples, 
                       replace_folder_test=replace_folder_test, seed=seed)
    else:
        raise ValueError("Invalid configuration/model input")