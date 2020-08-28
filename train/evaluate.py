import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *
import logging

import time
import os
from os import path
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model, Model
import matplotlib as mpl
mpl.use('Agg')

# Evaluate
# 1. Save results: points, predictions (optional), targets, energy, event info and telescope info
# 2. Calculate regression
# 3. Calculate resolution (angular or energy)
# 4. Prediction Samples (optinal)

def evaluate_experiment_folder(experiment_folder, save_predictions=True, save_samples=True, epoch=None, replace_folder_test=None, seed=None):
    """"
    Evaluate best model using experiment configuration.
    
    Parameters
    ==========
    experiment_folder :  `str`
        Path to experiment folder.
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
                 assembler=None,
                 save_predictions=True, save_samples=True,
                 model_name=None, replace_folder_test=None, seed=None):
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
    assembler : `gerumo.Assembler`, optional
        Assembler model instance.
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
    telescope = config["telescope"] 
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
    describe_dataset(test_dataset)
    if save_samples:
        # events with observations of every type of telescopes
        sample_events = [e for e, df in test_dataset.groupby("event_unique_id") if df["type"].nunique() == len(TELESCOPES)]
        # TODO: add custom seed
        r = np.random.RandomState(42)
        sample_events = r.choice(sample_events, size=5, replace=False)
        sample_dataset = test_dataset[test_dataset["event_unique_id"].isin(sample_events)]
        sample_dataset = filter_dataset(sample_dataset, telescope, [0], target_domains)

        print("Sample dataset")
        describe_dataset(sample_dataset)
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
    elif ("MultiCameraPipe" in preprocessing_parameters) and (telescope in preprocessing_parameters["MultiCameraPipe"]):
        camera_parameters = preprocessing_parameters["MultiCameraPipe"][telescope]
        camera_pipe = CameraPipe(telescope_type=telescope, version=version, **camera_parameters)
        preprocess_input_pipes['CameraPipe'] = camera_pipe
        
    if "TelescopeFeaturesPipe" in preprocessing_parameters:
        telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
        telescope_features_pipe = TelescopeFeaturesPipe(telescope_type=telescope, version=version, **telescopefeatures_parameters)
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
    ## 0. Evaluate with generator

    # Prepare targets points
    targets_values = []
    event_ids = []
    telescopes_ids = []
    telescopes_types = telescope
    true_energy = []
    predictions = []
    predictions_points = []

    if save_predictions:
        # Create predictions folder
        predictions_subfolder = path.join(output_folder, 'predictions')
        print("Saving predictions in:", predictions_subfolder)
        os.makedirs(predictions_subfolder, exist_ok=True)


    for batch_x, batch_t, batch_meta in tqdm(test_generator):
        # Update records
        event_ids.extend(batch_meta["event_id"])
        telescopes_ids.extend(batch_meta["telescope_id"])
        true_energy.extend(batch_meta["true_energy"])
        targets_values.extend(batch_t)

        # Predictions
        batch_p = assembler.exec_model_estimation(batch_x, model)
        batch_p_points = assembler.point_estimation(batch_p)
        predictions_points.extend(batch_p_points)
        
        # 0. Save predictions
        if save_predictions:
            predictions.extend(assembler.save_predictions(batch_p, predictions_subfolder))
    
    # Save model results
    results = pd.DataFrame({
        "event_id":         event_ids,
        "telescope_id":     telescopes_ids,
        "telescope_type":   [telescope]*len(telescopes_ids),
        "true_mc_energy":   true_energy,
    })
    targets_values = np.array(targets_values)
    predictions_points =  np.array(predictions_points)
    for i, target in enumerate(targets):
        results[f"true_{target}"] = targets_values[:,i].flatten()
        results[f"pred_{target}"] = predictions_points[:,i].flatten()
    
    if save_predictions:
        results[f"predictions"] = predictions
    results = results.dropna()

    # Free memory
    del targets_values
    del event_ids
    del predictions
    del predictions_points
    del true_energy
    del telescopes_ids
    
    # 1. Save results: points, targets, energy, event info and telescope info
    results.to_csv(path.join(output_folder, "results.csv"), index=False)
    
    # 2. Save samples plots
    if save_samples:
        # makedir
        samples_subfolder = path.join(output_folder, 'samples')
        print("Saving samples in:", samples_subfolder)
        os.makedirs(samples_subfolder, exist_ok=True)
        for batch_x, batch_t, batch_meta in tqdm(sample_generator):
            # Predictions
            batch_p = assembler.exec_model_estimation(batch_x, model)
            batch_p_points = assembler.point_estimation(batch_p)
            batch_input_image, batch_input_features = batch_x
            batch_iterator = zip(batch_input_image, batch_input_features, batch_p, batch_p_points, batch_t, batch_meta["event_id"], batch_meta["telescope_id"])
            for input_image_sample, input_features_sample, prediction, prediction_point, target_sample, event_id, telescope_id in batch_iterator:
                # save input
                image_filepath = path.join(samples_subfolder, f"event_id_{event_id}_telescope_id_{telescope_id}_input.png")
                plot_input_sample(input_image_sample, input_image_mode, input_features_sample, title=(event_id, telescope_id), make_simple=True,
                                    save_to=image_filepath)
                # save prediction
                prediction_filepath = path.join(samples_subfolder, f"event_id_{event_id}_telescope_id_{telescope_id}_prediction.png")
                plot_prediction(prediction, prediction_point, targets,
                                target_domains, target_resolutions, 
                                (event_id, telescope_id), target_sample, save_to=prediction_filepath)

    ## 3. Calculate regression
    print("Regression plots")
    scores = r2_score(results[[f"true_{target}" for target in targets]], results[[f"pred_{target}" for target in targets]], multioutput="raw_values")
    plot_regression_evaluation(results, targets, scores, save_to=path.join(output_folder, "regression.png"))

    if ("alt" in targets) and ("az" in targets) :
        print("Angular Reconstruction")
        ## 3. Calculate resolution (angular)
        plot_error_and_angular_resolution(results, save_to=path.join(output_folder, "angular_resolution.png"))
        
    if "log10_mc_energy" in targets:
        print("Energy resolution")
        ## 3. Calculate resolution (energy)
        plot_error_and_energy_resolution(results, save_to=path.join(output_folder, "energy_resolution.png"))
    
    if save_predictions:
        return results, predictions

    return results

def evaluate_assembler(assembler_config_file,
                       save_unit_evaluations=True, save_predictions=True, save_samples=True,
                       epoch=None, seed=None):
    # Load configuration file

    # Load experiment or models

    # Create output folder

    # Load Generator

    # Evaluate
    ## 1. Save results: points, predictions, targets, energy, event info and telescope info
    ## 2. Calculate regression
    ## 3. Calculate resolution (angular or energy)
    ## 4. Prediction Samples
    
    # Save Plots
    pass

# def evaluate(model_name, assembler_constructor, telescopes, evaluation_config, 
#              test_events_csv, test_telescope_csv, version, replace_folder_test, 
#              output_folder, min_observations,
#              input_image_mode, input_image_mask, input_features,
#              target_mode, targets, target_mode_config, target_domains):
    
#     # Generate new result folder
#     model_folder = path.join(output_folder, model_name)
#     os.makedirs(model_folder, exist_ok = True) 

#     # Prepare datasets
#     test_dataset = load_dataset(test_events_csv, test_telescope_csv, replace_folder_test)
#     test_dataset = aggregate_dataset(test_dataset, az=True, log10_mc_energy=True)
#     test_dataset = filter_dataset(test_dataset, telescopes, min_observations, target_domains)

#     # Preprocessing pipes
#     preprocess_input_pipes = []
#     preprocess_output_pipes = []

#     # Generators
#     batch_size = 16
#     telescope_types = [t for t in telescopes.keys() if telescopes[t] is not None]
    
#     # Test generators
#     test_generator =    AssemblerGenerator(
#                             test_dataset, telescope_types,
#                             batch_size, 
#                             input_image_mode=input_image_mode, 
#                             input_image_mask=input_image_mask, 
#                             input_features=input_features,
#                             targets=targets,
#                             target_mode=target_mode, 
#                             target_mode_config=target_mode_config,
#                             preprocess_input_pipes=preprocess_input_pipes,
#                             preprocess_output_pipes=preprocess_output_pipes,
#                             include_event_id=True,
#                             include_true_energy=True,
#                             version=version
#     )

#     # Sample Generator
#     small_size = 256
#     np.random.seed(evaluation_config["seed"]) 
#     sample_events = np.random.choice(test_dataset.event_unique_id.unique(), small_size)

#     # Models
#     sst = telescopes.get("SST1M_DigiCam", None)
#     mst = telescopes.get("MST_FlashCam", None)
#     lst = telescopes.get("LST_LSTCam", None)

#     assembler = assembler_constructor(
#                 sst1m_model_or_path=sst,
#                 mst_model_or_path=mst,
#                 lst_model_or_path=lst,
#                 targets=targets, 
#                 target_shapes=target_mode_config["target_shapes"],
#                 target_domains=target_mode_config["target_domains"],
#                 target_resolutions=target_mode_config["target_resolutions"],
#                 assembler_mode="normalized_product",
#                 point_estimation_mode="expected_value"
#     )
#     mode = assembler.mode
#     models_results = {}

#     # Units evaluation
#     for telescope_i, model in assembler.models.items():
#         if model is None or evaluation_config[telescope_i]["skip"]:
#             continue

#         # Telescope folder
#         telescope_folder = path.join(model_folder, telescope_i)
#         os.makedirs(telescope_folder, exist_ok = True) 

#         # Evaluation config
#         metrics_plot       = evaluation_config[telescope_i]["metrics_plot"]
#         save_df = evaluation_config[telescope_i]["predictions_points"]
#         probability_plot = evaluation_config[telescope_i]["probability_plot"] # no|sample
#         predictions_raw = evaluation_config[telescope_i]["predictions_raw"] # no|sample

#         # Telescope Sample Dataset
#         test_dataset_telescope = filter_dataset(test_dataset, [telescope_i], min_observations, target_domains)
#         bs =  1 if 0 < len(test_dataset_telescope) < batch_size else batch_size
#         telescope_generator =   AssemblerUnitGenerator(
#                                     test_dataset_telescope, 
#                                     batch_size=bs, 
#                                     input_image_mode=input_image_mode, 
#                                     input_image_mask=input_image_mask, 
#                                     input_features=input_features,
#                                     targets=targets,
#                                     target_mode="lineal",
#                                     target_mode_config=target_mode_config,
#                                     preprocess_input_pipes=preprocess_input_pipes,
#                                     preprocess_output_pipes=preprocess_output_pipes,
#                                     include_event_id=True,
#                                     include_true_energy=True,
#                                     version=version
#                                 )
#         # Prepare targets points
#         targets_values = []
#         event_ids = []
#         telescopes_ids = []
#         telescopes_types = telescope_i
#         true_energy = []
#         predictions = []
#         predictions_points = []

#         for x, t, meta in tqdm(telescope_generator):
#             # Update records
#             event_ids.extend(meta["event_id"])
#             telescopes_ids.extend(meta["telescope_id"])
#             true_energy.extend(meta["true_energy"])
#             targets_values.extend(t)

#             # Predictions
#             p = assembler.model_estimation(x, telescope_i, 0)
#             p_points = assembler.point_estimation(p)
#             predictions_points.extend(p_points)

#             # Save raw predictions
#             for prediction, prediction_point, target, event_id, telescope_id in zip(p, p_points, t, meta["event_id"], meta["telescope_id"]):
#                 # Save raw predictions
#                 if predictions_raw == "sample" and event_id in sample_events:
#                     # Predictions raw folder
#                     prob_folder = path.join(telescope_folder, "prob")
#                     os.makedirs(prob_folder, exist_ok = True) 
#                     # Save prediction
#                     save_path = path.join(prob_folder, f"{event_id}_{telescope_id}.npy")
#                     np.save(save_path, prediction)

#                 # Save probability plots
#                 if probability_plot == "sample" and event_id in sample_events:
#                     # Prob plots folder
#                     plot_folder = path.join(telescope_folder, "prob_plots")
#                     os.makedirs(plot_folder, exist_ok = True) 
#                     save_path = path.join(plot_folder, f"{event_id}_{telescope_id}.png")
                    
#                     plot_prediction(prediction, prediction_point, targets, 
#                                             assembler.target_domains, assembler.target_resolutions, 
#                                             (event_id, telescope_id),
#                                             target, save_to=save_path)

#         # Save model results
#         results = {
#             "event_id":     np.array(event_ids),
#             "telescope_id": np.array(telescopes_ids),
#             "telescope_type": np.array([telescope_i]*len(telescopes_ids)),
#             "predictions":  np.array(predictions_points), 
#             "targets":      np.array(targets_values), 
#             "true_energy":  np.array(true_energy), 
#         }
#         models_results[telescope_i] = results

#         # Free memory
#         del targets_values
#         del event_ids
#         del predictions_points
#         del true_energy
#         del telescopes_ids

#         if mode == "angular reconstruction":
#             df_ = pd.DataFrame({
#                 "event_id":         results["event_id"],
#                 "telescope_id":     results["telescope_id"],
#                 "telescope_type":   results["telescope_type"],
#                 "true_alt":         results["targets"][:,0].flatten(),
#                 "true_az":          results["targets"][:,1].flatten(),
#                 "pred_alt":         results["predictions"][:,0].flatten(),
#                 "pred_az":          results["predictions"][:,1].flatten(),
#                 "true_mc_energy":   results["true_energy"],
#             })
#             df_ = df_.dropna()
#             if metrics_plot:
#                 scores = r2_score(df_[["true_alt", "true_az"]], df_[["pred_alt", "pred_az"]], multioutput="raw_values")
#                 plot_regression_evaluation(df_, targets, scores, save_to=path.join(telescope_folder, "regression.png"))
#                 plot_error_and_angular_resolution(df_, save_to=path.join(telescope_folder, "angular_distribution.png"))
#             if save_df:
#                 df_.to_csv(path.join(telescope_folder, "predictions.csv"), index=False)
            
#         elif mode == "energy reconstruction":
#             df_ = pd.DataFrame({
#                 "event_id":             results["event_id"],
#                 "telescope_id":         results["telescope_id"],
#                 "telescope_type":       results["telescope_type"],
#                 "true_log10_mc_energy": results["targets"].flatten(),
#                 "pred_log10_mc_energy": results["predictions"].flatten(),
#                 "true_mc_energy":       results["true_energy"].flatten(),
#             })
#             df_ = df_.dropna()
#             if metrics_plot:
#                 scores = r2_score(df_[["true_log10_mc_energy"]], df_[["pred_log10_mc_energy"]], multioutput="raw_values")
#                 plot_regression_evaluation(df_, targets, scores, save_to=path.join(telescope_folder, "regression.png"))
#                 plot_error_and_energy_resolution(df_, save_to=path.join(telescope_folder, "energy_distribution.png"))
#             if save_df:
#                 df_.to_csv(path.join(telescope_folder, "predictions.csv"), index=False)

#         elif mode == "complete reconstruction":
#             raise NotImplemented
#         else:
#             raise NotImplemented
    
#     # Assembled evaluation
#     if evaluation_config["Assembler"]["skip"]:
#         return

#     # Assembler folder
#     assembler_folder = path.join(model_folder, "assembler")
#     os.makedirs(assembler_folder, exist_ok = True) 

#     # Evaluation config
#     metrics_plot       = evaluation_config["Assembler"]["metrics_plot"]
#     save_df = evaluation_config["Assembler"]["predictions_points"]
#     probability_plot = evaluation_config["Assembler"]["probability_plot"]   # no|sample
#     predictions_raw = evaluation_config["Assembler"]["predictions_raw"]     # no|sample

#     # evaluate
#     if predictions_raw == "no" and probability_plot == "no":
#         results = assembler.evaluate(test_generator)
#     else:
#         results, predictions = assembler.evaluate(test_generator, return_event_predictions=sample_events)

#     # Save raw results and plots
#     if predictions_raw != "no":
#         # Predictions raw folder
#         prob_folder = path.join(assembler_folder, "prob")
#         os.makedirs(prob_folder, exist_ok = True) 
#         for prediction, prediction_point, target_point, event_id in zip(predictions, predictions_points, targets_values, event_ids):
#             # Save prediction
#             save_path = path.join(prob_folder, f"{event_id}.npy")
#             np.save(save_path, prediction)

#     if probability_plot != "no":
#         # Prob plots folder
#         plot_folder = path.join(assembler_folder, "prob_plots")
#         os.makedirs(plot_folder, exist_ok = True) 
#         for prediction, prediction_point, target_point, event_id in zip(predictions, predictions_points, targets_values, event_ids):
#             save_path = path.join(plot_folder, f"{event_id}.png")
#             plot_prediction(prediction, prediction_point, targets, 
#                                     assembler.target_domains, assembler.target_resolution, 
#                                     event_id, target_point, save_to=save_path)

#     if mode == "angular reconstruction":
#         df_ = pd.DataFrame({
#             "event_id":         results["event_id"],
#             "true_alt":         results["targets"][:,0].flatten(),
#             "true_az":          results["targets"][:,1].flatten(),
#             "pred_alt":         results["predictions"][:,0].flatten(),
#             "pred_az":          results["predictions"][:,1].flatten(),
#             "true_mc_energy":   results["true_energy"],
#         })
#         df_ = df_.dropna()
#         if metrics_plot:
#             scores = r2_score(df_[["true_alt", "true_az"]], df_[["pred_alt", "pred_az"]], multioutput="raw_values")
#             plot_regression_evaluation(df_, targets, scores, save_to=path.join(assembler_folder, "regression.png"))
#             plot_error_and_angular_resolution(df_, save_to=path.join(assembler_folder, "angular_distribution.png"))
#         if save_df:
#             df_.to_csv(path.join(assembler_folder, "predictions.csv"), index=False)
            
#     elif mode == "energy reconstruction":
#         df_ = pd.DataFrame({
#             "event_id":             results["event_id"],
#             "true_log10_mc_energy": results["targets"].flatten(),
#             "pred_log10_mc_energy": results["predictions"].flatten(),
#             "true_mc_energy":       results["true_energy"].flatten(),
#         })
#         df_ = df_.dropna()
#         if metrics_plot:
#             scores = r2_score(df_[["true_log10_mc_energy"]], df_[["pred_log10_mc_energy"]], multioutput="raw_values")
#             plot_regression_evaluation(df_, targets, scores, save_to=path.join(assembler_folder, "regression.png"))
#             plot_error_and_energy_resolution(df_, save_to=path.join(assembler_folder, "energy_distribution.png"))
#         if save_df:
#             df_.to_csv(path.join(assembler_folder, "predictions.csv"), index=False)

#     elif mode == "complete reconstruction":
#         raise NotImplemented
#     else:
#         raise NotImplemented




if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Evaluate models.")
    ap.add_argument("-a", "--assembler", type=str, default=None, 
                    help="Assembler configuration file.")
    ap.add_argument("-e", "--experiment", type=str, default=None, 
                    help="Experiment folder.")
    ap.add_argument("-m", "--model", type=str, default=None, 
                    help="Model checkpoint. (require configuration file)")
    ap.add_argument("-c", "--config", type=str, default=None, 
                    help="Model configuration file. (require model checkpoint)")
    ap.add_argument("--samples", action="store_true", dest="save_samples",
                     help="Save inputs and predictions plots from sample dataset.")
    ap.add_argument("--predictions", action="store_true", dest="save_predictions",
                     help="Save predictions from test dataset.")
    ap.add_argument("--seed", type=int, default=None, 
                    help="Set random state seed.")
    ap.add_argument("--epoch", type=int, default=None, 
                    help="Select epoch from experiment folder.")
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
    epoch = args["epoch"]
    seed = args["seed"]

    if assembler_config is not None:
        raise NotImplementedError
    elif experiment_folder is not None:
        evaluate_experiment_folder( experiment_folder, 
                                    save_predictions=save_predictions, save_samples=save_samples,
                                    epoch=epoch, replace_folder_test=replace_folder_test, seed=seed)
    elif (model_config is not None) and (model_checkpoint is not None):
        raise NotImplementedError
    else:
        raise ValueError("Invalid configuration/model input")

    """
    config_file = args["config"]
    
    if config_file is None:
        print("No config file")
        exit(1)
    else:
        print(f"Loading config from: {config_file}")
    with open(config_file) as cfg_file:
        config = json.load(cfg_file)

    # Model
    model_name = config["model_name"]
    assembler_constructor = ASSEMBLERS[config["assembler_constructor"]]
    
    # Dataset Parameters
    output_folder = config["output_folder"]
    replace_folder_test = config["replace_folder_test"]
    test_events_csv    = config["test_events_csv"]
    test_telescope_csv = config["test_telescope_csv"]
    version = config["version"]

    # Input and Target Parameters 
    telescopes = config["telescopes"]
    min_observations = config["min_observations"]
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    input_features = config["input_features"]
    targets = config["targets"]
    target_mode = "lineal"
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    if  config["assembler_constructor"] == 'umonna':
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
            "target_shapes":      tuple([np.inf      for target in targets]),
            "target_resolutions": tuple([np.inf      for target in targets])
        }
        target_resolutions = tuple([np.inf      for target in targets])

    # Evaluation Parameters
    evaluation_config = config["evaluation"]

    evaluate(
        model_name, assembler_constructor, telescopes, evaluation_config,
        test_events_csv, test_telescope_csv, version, replace_folder_test, 
        output_folder, min_observations,
        input_image_mode, input_image_mask, input_features,
        target_mode, targets, target_mode_config, target_domains
    )   
    """