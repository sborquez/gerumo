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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def evaluate(model_name, assembler_constructor, telescopes, evaluation_config, 
             test_events_csv, test_telescope_csv, version, replace_folder_test, 
             output_folder, min_observations,
             input_image_mode, input_image_mask, input_features,
             target_mode, targets, target_mode_config, target_domains, preprocessing_parameters):
    
    # Generate new result folder
    model_folder = path.join(output_folder, model_name)
    os.makedirs(model_folder, exist_ok = True) 

    # Prepare datasets
    test_dataset = load_dataset(test_events_csv, test_telescope_csv, replace_folder_test)
    test_dataset = aggregate_dataset(test_dataset, az=True, log10_mc_energy=True)
    test_dataset = filter_dataset(test_dataset, telescopes, min_observations, target_domains)

    # Preprocessing pipes
    preprocess_input_pipes = {}
    ## output preprocessing
    preprocess_output_pipes = {}
    
    # Generators
    batch_size = 16
    telescope_types = [t for t in telescopes.keys() if telescopes[t] is not None]
    
   
    # Sample Generator
    small_size = 256
    np.random.seed(evaluation_config["seed"]) 
    sample_events = np.random.choice(test_dataset.event_unique_id.unique(), small_size)

    # Model paths
    sst = telescopes.get("SST1M_DigiCam", None)
    mst = telescopes.get("MST_FlashCam", None)
    lst = telescopes.get("LST_LSTCam", None)

    #model generation
    #assembler_constructor = BMO_DET, from the corresponding assembler_config file
    #for single and stereo predictions
    assembler = assembler_constructor(
                sst1m_model_or_path=sst,
                mst_model_or_path=mst,
                lst_model_or_path=lst,
                targets=targets, 
                target_shapes=target_mode_config["target_shapes"],
                target_domains=target_mode_config["target_domains"],
                target_resolutions=target_mode_config["target_resolutions"],
                #assembler_mode = "resample",  
                assembler_mode="normalized_product",
                point_estimation_mode="expected_value"
    )

    mode = assembler.mode
    models_results = {}

    # Units evaluation
    # assembler.models is obtained from the original definition of assembler constructor, which is
    # given in models/assembler.py. In particular, telescopes and models are updated when load_model is called
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

        # BP: short the test for rapid visualization
        #test_size = 100
        #test_dataset_telescope = test_dataset_telescope.iloc[0:test_size,:]
        
        bs =  1 if 0 < len(test_dataset_telescope) < batch_size else batch_size
        print(batch_size, len(test_dataset_telescope))
       
        #data generation, telescope_generator elements are batches of inputs
        #the class AssemblerUnitGenerator is defined in data/generator.py
        if "CameraPipe" in preprocessing_parameters:
            camera_parameters = preprocessing_parameters["CameraPipe"]
            camera_pipe = CameraPipe(telescope_type=telescope_i, version=version, **camera_parameters)
            preprocess_input_pipes['CameraPipe'] = camera_pipe
            
        if "TelescopeFeaturesPipe" in preprocessing_parameters:
            telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
            telescope_features_pipe = TelescopeFeaturesPipe(telescope_type=telescope_i, version=version, \
                                                            **telescopefeatures_parameters)
            preprocess_input_pipes['TelescopeFeaturesPipe'] = telescope_features_pipe
        
        telescope_generator =   AssemblerUnitGenerator(
                                    test_dataset_telescope, 
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
                                    include_true_energy=True,
                                    version=version
                                )

        #BP: mono batch predictions, loss checks, plots generation
        mae_con = 0
        mae_sum = 0

        pred = np.zeros((len(test_dataset_telescope), len(targets)))
        target = np.zeros((len(test_dataset_telescope), len(targets)))
        true_energy = np.zeros((len(test_dataset_telescope)))

        local_path = f"/home/bapanes/Research-Now/local/ml-valpo-local/umonna/dataset/ML1/models_lc/"
        
        for x, target_array, meta in tqdm(telescope_generator):
            # Predictions

            #x[0] contains the batch of 16 images of shape (2,84,29,3)
            #x[1] contains the batch of 16 external parameters of shape (2)
            #print(x[0].shape)
            #print(x[1].shape)

            #save the first 10 inputs
            if mae_con == 0:

                print(x[0].shape)
                print(x[1].shape)
            
                for con_plot in range(10):
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
                    ax1.set_title("Charge")
                    divider = make_axes_locatable(ax1)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    color_map = ax1.imshow(x[0][con_plot,0,:,:,0])
                    plt.colorbar(color_map, cax = cax)

                    ax2.set_title("Peak")
                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    color_map = ax2.imshow(x[0][con_plot,0,:,:,1])
                    plt.colorbar(color_map, cax=cax)

                    plt.savefig(f"{local_path}/event_{con_plot}.png")
                    plt.close()
                
            #x[0] = np.random.random_sample((16, 2, 84, 29, 3))

            ###########################
            #Prediction part
            ###########################
            
            #cnn_det
            batch_pred = assembler.model_estimation(x, telescope_i, 0)

            #save the information in arrays
            pred_array = np.array(batch_pred)
            target_array = np.array(target_array)
            true_energy_array = np.array(meta['true_energy'])

            #fill the full arrays
            pred[mae_con*batch_size:(mae_con+1)*batch_size,:] = pred_array
            target[mae_con*batch_size:(mae_con+1)*batch_size,:] = target_array
            true_energy[mae_con*batch_size:(mae_con+1)*batch_size] = true_energy_array

            mae_sum = mae_sum + mean_absolute_error(pred_array, target_array)
            
            mae_con = mae_con + 1
            #print("partial mae loss:", mean_absolute_error(pred_array, target_array)) 
        
        for con in range(20):
            print(true_energy[con], pred[con], target[con])

        #mean mae value
        mae_average = mae_sum/mae_con
        print("average loss: ", mae_average)

        #angular resolution plot
        predicted_az = pred[:,0]
        predicted_alt = pred[:,1]
        
        true_az = target[:,0]
        true_alt = target[:,1]

        true_mc_energy = true_energy[:]

        #energy histogram
        logbins = np.logspace(np.log10(0.01), np.log10(100), 10)
        
        plt.hist(true_mc_energy, bins = logbins)
        plt.xscale('log')
        plt.xlabel("True energy [TeV]")
        plt.ylabel("Counts")
        plt.savefig(f"{local_path}/true_energy_{mae_average:.4f}.png")
        plt.close()
        
        # Create Figure and axis
        fig, axis = plt.subplots(1, 2, figsize=(14, 6))
    
        # Style
        plt.suptitle("Angular Reconstruction")

        # Generate two plots
        show_absolute_error_angular(predicted_alt, predicted_az, true_alt, true_az,
                                    bias_correction=None, ax=axis[0], bins=80, 
                                    percentile_plot_range=80)

        show_angular_resolution(predicted_alt, predicted_az, true_alt, true_az, true_mc_energy,
                                percentile=68.27, confidence_level=0.95, bias_correction=False,
                                label="MST mono", xlim=None, ylim=None, ax=axis[1])
    
        # Save 
        plt.savefig(f"{local_path}/angular_resolution_{mae_average:.4f}.png")
        plt.close()
                           
        #cnn_det plots
        plt.scatter(target[:,0], pred[:,0], s=10, color='blue', alpha=0.5)
        plt.title('Evaluation of CNN-DET-ADAM-MAE predictions on test set')
        plt.xlabel('target az')
        plt.ylabel('predicted az')
        plt.xlim(-0.52,0.52)
        plt.ylim(-0.52,0.52)
        plt.savefig(f"{local_path}/scatter_az_cnn_det_adam_mae_{mae_average:.4f}.png")

        plt.scatter(target[:,1], pred[:,1], s=10, color='blue', alpha=0.5)
        plt.title('Evaluation of CNN-DET-ADAM-MAE predictions on test set')
        plt.xlabel('target alt')
        plt.ylabel('predicted alt')
        plt.xlim(1.05, 1.382)
        plt.ylim(1.05, 1.382)
        plt.savefig(f"{local_path}/scatter_alt_cnn_det_adam_mae_{mae_average:.4f}.png")

        return 0
        
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
    # Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})
    
    evaluate(
        model_name, assembler_constructor, telescopes, evaluation_config,
        test_events_csv, test_telescope_csv, version, replace_folder_test, 
        output_folder, min_observations,
        input_image_mode, input_image_mask, input_features,
        target_mode, targets, target_mode_config, target_domains, preprocessing_parameters
    )   
