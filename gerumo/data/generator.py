"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""

from tensorflow import keras
import numpy as np
from . import load_cameras, cameras_to_images, targets_to_matrix


class AssemblerUnitGenerator(keras.utils.Sequence):
    """
    AssemblerUnitGenerator
    """

    def __init__(self, dataset, batch_size, 
                 input_image_mode, 
                 input_image_mask,
                 input_features,
                 targets,
                 target_mode,
                 target_mode_config,
                 preprocess_input_pipes=[],
                 preprocess_output_pipes=[],
                 include_event_id=False,
                 include_true_energy=False,
                 shuffle=False):

        # Dataset with one telescope type
        self.dataset = dataset
        self.telescope_type = dataset.iloc[0].type

        # How to generate input image
        self.input_image_mode = input_image_mode
        self.input_image_mask = input_image_mask
        # Which extra telescope feature use
        self.input_features = input_features
        # Add normalization and stardarization 
        self.preprocess_input_pipes = preprocess_input_pipes

        # Which target predict
        self.targets = targets
        # How to generate target matrix
        self.target_mode = target_mode
        self.target_mode_config = target_mode_config
        # Add normalization and stardarization 
        self.preprocess_output_pipes = preprocess_output_pipes

        # Generator parameters
        self.batch_size = batch_size
        self.size = len(self.dataset)
        self.shuffle = shuffle
        self.include_event_id = include_event_id
        self.include_true_energy = include_true_energy

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        # This is for CTA evaluation metrics
        if self.include_event_id or self.include_true_energy:
            meta = self.__meta_generation(indexes)
            return X, y, meta
        
        else:
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        # TODO: Add shuffle but without mix hdf5 files
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples'
        # Initialization
        """
        X : [images, features]
        # images : (n_samples, *dim, n_channels)
        # features : (n_samples, n_features)

        y : (n_samples, *target_dim)
        """
        batch_dataset = self.dataset.iloc[list_indexes]
        # dataset contains only one telescope type
        telescope_types = [self.telescope_type]*len(batch_dataset) 
        images = cameras_to_images(load_cameras(batch_dataset), telescope_types, 
                                   self.input_image_mode, self.input_image_mask)
        batch_images = np.array(images)                                   
        batch_telescope_features = batch_dataset[self.input_features].values
        # TODO: Add preprocessing steps
        X = [batch_images, batch_telescope_features]

        if self.target_mode is not None:
            y = targets_to_matrix(targets_values=batch_dataset[self.targets].values, 
                              target_names=self.targets,
                              target_mode=self.target_mode, 
                              target_mode_config=self.target_mode_config)
            # TODO: Add preprocessing steps
        else:
            y = None

        return X, y

    def __meta_generation(self, list_indexes):
        'Generates metadata of the batch samples'
        # Initialization
        """
        meta : {
            "event_id"    : event id, for identify each plot
            "telescope_id": telescope id, for identify each telescope
            "telescope_type": for identify each type
            "true_energy" : value of mc_energy in TeV for ploting angular resolution
        }
        """
        meta = {}
        batch_dataset = self.dataset.iloc[list_indexes]

        if self.include_event_id:
            meta["telescope_id"] = batch_dataset.telescope_id.to_numpy()
            meta["event_id"] = batch_dataset.event_unique_id.to_numpy()
            meta["type"]    = batch_dataset["type"].to_numpy()
        if self.include_true_energy:
            meta["true_energy"] = batch_dataset.mc_energy.to_numpy()

        return meta


class AssemblerGenerator(keras.utils.Sequence):
    """
    AssemblerGenerator

    Load data used for an Assembler with one or more telescopes.

    Parameters
    ==========
    dataset : pd.DataFrame
        observarion dataset. Must contains observartion for all telescopes
        in `telescopes` dictionary
    telescopes : `list` of `str` or 'str'
        Selected telescopes type for the dataset. 'str' if is just one.
    number_of_observations : `list` of `int` or 'int
        For each telescope in 'telescopes' parameter, the minimum amount
        of observations. 'int' if is just one.
    domain : `dict` [ `str`,  `tuple` of `int`]
        A dictionary with names of columns and their value range.
    Returns
    =======
     `pd.DataFrame`
        Filtered dataset.

    """

    def __init__(self, dataset, telescopes, batch_size, 
                 input_image_mode, 
                 input_image_mask,
                 input_features,
                 targets,
                 target_mode,
                 target_mode_config,
                 preprocess_input_pipes={},
                 preprocess_output_pipes={},
                 include_event_id=False,
                 include_true_energy=False,
                 shuffle=False):

        if isinstance(telescopes, str):
            telescopes = [telescopes]

        # Dataset with all telescope type ing telescope_types
        self.dataset = dataset.groupby("event_unique_id")
        self.events = np.array(list(self.dataset.groups.keys()))
        self.telescope_types = sorted(telescopes)

        # How to generate input image
        self.input_image_mode = input_image_mode
        self.input_image_mask = input_image_mask
        # Which extra telescope feature use
        self.input_features = input_features
        # Add normalization and stardarization 
        self.preprocess_input_pipes = preprocess_input_pipes

        # Which target predict
        self.targets = targets
        # How to generate target matrix
        self.target_mode = target_mode
        self.target_mode_config = target_mode_config
        # Add normalization and stardarization 
        self.preprocess_output_pipes = preprocess_output_pipes

        # Generator parameters
        self.batch_size = batch_size
        self.size = len(self.dataset)
        self.shuffle = shuffle
        self.include_event_id = include_event_id
        self.include_true_energy = include_true_energy

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if 0 < len(self.dataset) < self.batch_size:
            return 1:
        else:
            return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        
        # This is for CTA evaluation metrics
        if self.include_event_id or self.include_true_energy:
            meta = self.__meta_generation(indexes)
            return X, y, meta
        
        else:
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        # TODO: Add shuffle but without mix hdf5 files
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples'
        # Initialization
        batch_events = self.events[list_indexes]
        X = []
        """
        X : [
                {
                    "tel1" : [ 
                        event_images_tel1_array, 
                        event_telescope_features_tel1_array
                    ],  # for each telescope
                }, # for each event
        ]
        # event_images_tel1_array : (n_samples, *dim, n_channels), 
        # event_telescope_features_tel1_array : (n_samples, n_features)
        
        y : (n_events, *target_dim)

        # optionals
        meta : {
            #optional
            "mc_energy: [ ... ],
            #optional
            "events_id: [ ... ]
        }
        """
        for event in batch_events:
            event_dataset = self.dataset.get_group(event).sort_values(by="type")
            # dataset contains only one telescope type
            telescope_types = list(event_dataset.type)

            event_images = cameras_to_images(load_cameras(event_dataset), telescope_types, 
                                   self.input_image_mode, self.input_image_mask)

            event_telescopes_features = event_dataset[self.input_features].values

            event_by_telescope = {}
            for telescope_type in self.telescope_types:
                telescope_indices = [i for i, t in enumerate(telescope_types) if t == telescope_type]
                event_images_by_telescope = np.array([event_images[t_i] for t_i in telescope_indices])
                event_telescopes_features_by_telescope = np.array([event_telescopes_features[t_i] for t_i in telescope_indices])
                # TODO: Add preprocessing steps
                event_by_telescope[telescope_type] = [ 
                    event_images_by_telescope, event_telescopes_features_by_telescope
                ]
            X.append(event_by_telescope)

        if self.target_mode is not None:
            # return the target of the first member of each grouped event
            events_targets = self.dataset.first()
            events_targets = events_targets[events_targets.index.isin(batch_events)][self.targets]
            y = targets_to_matrix(targets_values=events_targets.values, 
                            target_names=self.targets,
                            target_mode=self.target_mode, 
                            target_mode_config=self.target_mode_config)
            # TODO: Add preprocessing steps
        else:
            y = None

        return X, y

    def __meta_generation(self, list_indexes):
        'Generates metadata of the batch samples'
        # Initialization
        """
        meta : {
            "event_id"    : event id, for identify each plot
            "true_energy" : value of mc_energy in TeV for ploting angular resolution
        }
        """
                # Initialization
        meta = {}
        batch_events = self.events[list_indexes]

        if self.include_event_id:
            meta["event_id"] = batch_events

        if self.include_true_energy:
            grouped_mc_energy = self.dataset.mc_energy
            meta["true_energy"] = np.array([grouped_mc_energy.get_group(event).iloc[0] for event in batch_events])

        return meta