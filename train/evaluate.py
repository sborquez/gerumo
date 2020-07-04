import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *

import logging
import time
import os
from os import path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def evaluate(model_name, assembler_constructor, telescopes, evaluation_config, 
             test_events_csv, test_telescope_csv, replace_folder_test, 
             output_folder, min_observations,
             input_image_mode, input_image_mask, input_features,
             target_mode, targets, target_mode_config, target_domains):
    
    # Generate new result folder
    model_folder = path.join(output_folder, model_name)
    os.makedirs(model_folder, exist_ok = True) 

    # Prepare datasets
    test_dataset = load_dataset(test_events_csv, test_telescope_csv, replace_folder_test)
    test_dataset = aggregate_dataset(test_dataset, az=True, log10_mc_energy=True)
    test_dataset = filter_dataset(test_dataset, telescopes, min_observations, target_domains)

    # Preprocessing pipes
    preprocess_input_pipes = []
    preprocess_output_pipes = []

    # Generators
    batch_size = 16
    telescope_types = [t for t in telescopes.keys() if telescopes[t] is not None]
    
    # Test generators
    test_generator = AssemblerGenerator(test_dataset, telescope_types,
                                        batch_size, 
                                        input_image_mode=input_image_mode, 
                                        input_image_mask=input_image_mask, 
                                        input_features=input_features,
                                        targets=targets,
                                        target_mode=target_mode, 
                                        target_mode_config=target_mode_config,
                                        preprocess_input_pipes=preprocess_input_pipes,
                                        preprocess_output_pipes=preprocess_output_pipes,
                                        include_event_id=True,
                                        include_true_energy=True
    )

    # Sample Generator
    small_size = 256
    np.random.seed(evaluation_config["seed"]) 
    sample_events = np.random.choice(test_dataset.event_unique_id.unique(), small_size)

    # Models
    sst = telescopes.get("SST1M_DigiCam", None)
    mst = telescopes.get("MST_FlashCam", None)
    lst = telescopes.get("LST_LSTCam", None)


    if target_mode_config["target_resolutions"] is not None:
        assembler = assembler_constructor(
                    sst1m_model_or_path=sst,
                    mst_model_or_path=mst,
                    lst_model_or_path=lst,
                    targets=targets, 
                    target_shapes=target_mode_config["target_shapes"],
                    target_domains=target_mode_config["target_domains"],
                    target_resolutions=target_mode_config["target_resolutions"],
                    assembler_mode="normalized_product",
                    point_estimation_mode="expected_value"
        )
    else:
        assembler = assembler_constructor(
                    sst1m_model_or_path=sst,
                    mst_model_or_path=mst,
                    lst_model_or_path=lst,
                    targets=targets, 
                    target_shapes=target_mode_config["target_shapes"],
                    target_domains=target_mode_config["target_domains"],
                    assembler_mode="normalized_product",
                    point_estimation_mode="expected_value"
        )
    mode = assembler.mode
    models_results = {}

    # Units evaluation
    for telescope_i, model in assembler.models.items():
        if model is None or evaluation_config[telescope_i]["skip"]:
            continue

        # Telescope folder
        telescope_folder = path.join(model_folder, telescope_i)
        os.makedirs(telescope_folder, exist_ok = True) 

        # Evaluation config
        metrics_plot       = evaluation_config[telescope_i]["metrics_plot"]
        save_df = evaluation_config[telescope_i]["predictions_points"]
        probability_plot = evaluation_config[telescope_i]["probability_plot"] # no|sample
        predictions_raw = evaluation_config[telescope_i]["predictions_raw"] # no|sample

        # Telescope Sample Dataset
        test_dataset_telescope = filter_dataset(test_dataset, [telescope_i], min_observations, target_domains)
        bs =  1 if 0 < len(test_dataset_telescope) < batch_size else batch_size
        telescope_generator = AssemblerUnitGenerator(test_dataset_telescope, 
                                                batch_size=bs, 
                                                input_image_mode=input_image_mode, 
                                                input_image_mask=input_image_mask, 
                                                input_features=input_features,
                                                targets=targets,
                                                target_mode="lineal",
                                                target_mode_config=target_mode_config,
                                                preprocess_input_pipes=preprocess_input_pipes,
                                                preprocess_output_pipes=preprocess_output_pipes,
                                                include_event_id=True,
                                                include_true_energy=True
                                                )
        # Prepare targets points
        targets_values = []
        event_ids = []
        telescopes_ids = []
        telescopes_types = telescope_i
        true_energy = []
        predictions = []
        predictions_points = []

        for x, t, meta in tqdm(telescope_generator):
            # Update records
            event_ids.extend(meta["event_id"])
            telescopes_ids.extend(meta["telescope_id"])
            true_energy.extend(meta["true_energy"])
            targets_values.extend(t)

            # Predictions
            p = assembler.model_estimation(x, telescope_i, 0)
            p_points = assembler.point_estimation(p)
            predictions_points.extend(p_points)

            # Save raw predictions
            for prediction, prediction_point, target, event_id, telescope_id in zip(p, p_points, t, meta["event_id"], meta["telescope_id"]):
                # Save raw predictions
                if predictions_raw == "sample" and event_id in sample_events:
                    # Predictions raw folder
                    prob_folder = path.join(telescope_folder, "prob")
                    os.makedirs(prob_folder, exist_ok = True) 
                    # Save prediction
                    save_path = path.join(prob_folder, f"{event_id}_{telescope_id}.npy")
                    np.save(save_path, prediction)

                # Save probability plots
                if probability_plot == "sample" and event_id in sample_events:
                    # Prob plots folder
                    plot_folder = path.join(telescope_folder, "prob_plots")
                    os.makedirs(plot_folder, exist_ok = True) 
                    save_path = path.join(plot_folder, f"{event_id}_{telescope_id}.png")
                    
                    plot_prediction(prediction, prediction_point, targets, 
                                            assembler.target_domains, assembler.target_resolutions, 
                                            (event_id, telescope_id),
                                            target, save_to=save_path)

        # Save model results
        results = {
            "event_id":     np.array(event_ids),
            "telescope_id": np.array(telescopes_ids),
            "telescope_type": np.array([telescope_i]*len(telescopes_ids)),
            "predictions":  np.array(predictions_points), 
            "targets":      np.array(targets_values), 
            "true_energy":  np.array(true_energy), 
        }
        models_results[telescope_i] = results

        # Free memory
        del targets_values
        del event_ids
        del predictions_points
        del true_energy
        del telescopes_ids

        if mode == "angular reconstruction":
            df_ = pd.DataFrame({
                "event_id":         results["event_id"],
                "telescope_id":     results["telescope_id"],
                "telescope_type":   results["telescope_type"],
                "true_alt":         results["targets"][:,0].flatten(),
                "true_az":          results["targets"][:,1].flatten(),
                "pred_alt":         results["predictions"][:,0].flatten(),
                "pred_az":          results["predictions"][:,1].flatten(),
                "true_mc_energy":   results["true_energy"],
            })
            df_ = df_.dropna()
            if metrics_plot:
                scores = r2_score(df_[["true_alt", "true_az"]], df_[["pred_alt", "pred_az"]], multioutput="raw_values")
                plot_regression_evaluation(df_, targets, scores, save_to=path.join(telescope_folder, "regression.png"))
                plot_error_and_angular_resolution(df_, save_to=path.join(telescope_folder, "angular_distribution.png"))
            if save_df:
                df_.to_csv(path.join(telescope_folder, "predictions.csv"), index=False)
            
        elif mode == "energy reconstruction":
            df_ = pd.DataFrame({
                "event_id":             results["event_id"],
                "telescope_id":         results["telescope_id"],
                "telescope_type":       results["telescope_type"],
                "true_log10_mc_energy": results["targets"].flatten(),
                "pred_log10_mc_energy": results["predictions"].flatten(),
                "true_mc_energy":       results["true_energy"].flatten(),
            })
            df_ = df_.dropna()
            if metrics_plot:
                scores = r2_score(df_[["true_log10_mc_energy"]], df_[["pred_log10_mc_energy"]], multioutput="raw_values")
                plot_regression_evaluation(df_, targets, scores, save_to=path.join(telescope_folder, "regression.png"))
                plot_error_and_energy_resolution(df_, save_to=path.join(telescope_folder, "energy_distribution.png"))
            if save_df:
                df_.to_csv(path.join(telescope_folder, "predictions.csv"), index=False)

        elif mode == "complete reconstruction":
            raise NotImplemented
        else:
            raise NotImplemented
    
    # Assembled evaluation
    if evaluation_config["Assembler"]["skip"]:
        return

    # Assembler folder
    assembler_folder = path.join(model_folder, "assembler")
    os.makedirs(assembler_folder, exist_ok = True) 

    # Evaluation config
    metrics_plot       = evaluation_config["Assembler"]["metrics_plot"]
    save_df = evaluation_config["Assembler"]["predictions_points"]
    probability_plot = evaluation_config["Assembler"]["probability_plot"]   # no|sample
    predictions_raw = evaluation_config["Assembler"]["predictions_raw"]     # no|sample

    # evaluate
    if predictions_raw == "no" and probability_plot == "no":
        results = assembler.evaluate(test_generator)
    else:
        results, predictions = assembler.evaluate(test_generator, return_event_predictions=sample_events)

    # Save raw results and plots
    if predictions_raw != "no":
        # Predictions raw folder
        prob_folder = path.join(assembler_folder, "prob")
        os.makedirs(prob_folder, exist_ok = True) 
        for prediction, prediction_point, target_point, event_id in zip(predictions, predictions_points, targets_values, event_ids):
            # Save prediction
            save_path = path.join(prob_folder, f"{event_id}.npy")
            np.save(save_path, prediction)

    if probability_plot != "no":
        # Prob plots folder
        plot_folder = path.join(assembler_folder, "prob_plots")
        os.makedirs(plot_folder, exist_ok = True) 
        for prediction, prediction_point, target_point, event_id in zip(predictions, predictions_points, targets_values, event_ids):
            save_path = path.join(plot_folder, f"{event_id}.png")
            plot_prediction(prediction, prediction_point, targets, 
                                    assembler.target_domains, assembler.target_resolution, 
                                    event_id, target_point, save_to=save_path)

    if mode == "angular reconstruction":
        df_ = pd.DataFrame({
            "event_id":         results["event_id"],
            "true_alt":         results["targets"][:,0].flatten(),
            "true_az":          results["targets"][:,1].flatten(),
            "pred_alt":         results["predictions"][:,0].flatten(),
            "pred_az":          results["predictions"][:,1].flatten(),
            "true_mc_energy":   results["true_energy"],
        })
        df_ = df_.dropna()
        if metrics_plot:
            scores = r2_score(df_[["true_alt", "true_az"]], df_[["pred_alt", "pred_az"]], multioutput="raw_values")
            plot_regression_evaluation(df_, targets, scores, save_to=path.join(assembler_folder, "regression.png"))
            plot_error_and_angular_resolution(df_, save_to=path.join(assembler_folder, "angular_distribution.png"))
        if save_df:
            df_.to_csv(path.join(assembler_folder, "predictions.csv"), index=False)
            
    elif mode == "energy reconstruction":
        df_ = pd.DataFrame({
            "event_id":             results["event_id"],
            "true_log10_mc_energy": results["targets"].flatten(),
            "pred_log10_mc_energy": results["predictions"].flatten(),
            "true_mc_energy":       results["true_energy"].flatten(),
        })
        df_ = df_.dropna()
        if metrics_plot:
            scores = r2_score(df_[["true_log10_mc_energy"]], df_[["pred_log10_mc_energy"]], multioutput="raw_values")
            plot_regression_evaluation(df_, targets, scores, save_to=path.join(assembler_folder, "regression.png"))
            plot_error_and_energy_resolution(df_, save_to=path.join(assembler_folder, "energy_distribution.png"))
        if save_df:
            df_.to_csv(path.join(assembler_folder, "predictions.csv"), index=False)

    elif mode == "complete reconstruction":
        raise NotImplemented
    else:
        raise NotImplemented




if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Evaluate models.")
    ap.add_argument("-c", "--config", type=str, default=None, help="Configuration file for model/experiment.")
    args = vars(ap.parse_args()) 
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

    # Input and Target Parameters 
    telescopes = config["telescopes"]
    min_observations = config["min_observations"]
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    input_features = config["input_features"]
    targets = config["targets"]
    target_mode = "lineal" if config["assembler_constructor"] != 'umonna' else config['target_mode']
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    if target_mode != 'lineal':
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
            "target_resolutions": None
        }

    # Evaluation Parameters
    evaluation_config = config["evaluation"]

    evaluate(
        model_name, telescopes, evaluation_config,
        test_events_csv, test_telescope_csv, replace_folder_test, 
        output_folder, min_observations,
        input_image_mode, input_image_mask, input_features,
        target_mode, targets, target_mode_config, target_domains
    )   