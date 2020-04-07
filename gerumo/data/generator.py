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
                 target_shape,
                 preprocess_input_pipes=[],
                 preprocess_output_pipes=[],
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
        self.target_shape = target_shape
        # Add normalization and stardarization 
        self.preprocess_output_pipes = preprocess_output_pipes

        # Generator parameters
        self.batch_size = batch_size
        self.size = len(self.dataset)
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

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
        # X : (n_samples, *dim, n_channels)

        batch_dataset = self.dataset.iloc[list_indexes]
        images = cameras_to_images(load_cameras(batch_dataset), [self.telescope_type]*len(batch_dataset), 
                                   self.input_image_mode, self.input_image_mask)
        batch_images = np.array(images)                                   
        batch_telescope_features = batch_dataset[self.input_features].values
        # TODO: Add preprocessing steps
        X = [batch_images, batch_telescope_features]

        y = targets_to_matrix(targets_values=batch_dataset[self.targets].values, 
                              target_names=self.targets,
                              target_mode=self.target_mode, 
                              target_mode_config=self.target_mode_config)
        # TODO: Add preprocessing steps

        return X, y


class AssemberGenerator(keras.utils.Sequence):
    pass