"""
CNN DET: Bayesian Multi Observer in Deterministic mode
======
Convolutional Neural Network for mono and multi-stereo event reconstruction
"""

import numpy as np
import scipy as sp
import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Lambda,
    Conv2D, MaxPooling2D, Conv3D,
    Conv2DTranspose, Conv3DTranspose,
    UpSampling1D, UpSampling2D, UpSampling3D,
    AveragePooling1D, AveragePooling2D, AveragePooling3D,
    Dense, Flatten, Concatenate, Reshape,
    Activation, BatchNormalization, Dropout
)
from tensorflow.keras.regularizers import l2
from . import CUSTOM_OBJECTS
from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax


def cnn_det_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                conv_kernel_sizes=[5, 3, 3], latent_variables=200, dense_layer_blocks=5, activity_regularizer_l2=None):
    """Build Deterministic CNN Unit Model
    Parameters
    ==========
        telescope
        image_mode
        image_mask
        input_img_shape
        input_features_shape
        output_mode
        output_shape
    Return
    ======
        keras.Model 
    """
    # Soport lineal only
    if target_mode != 'lineal':
        raise ValueError(f"Invalid target_mode: '{target_mode}'" )

    # Image Encoding Block
    ## HexConvLayer
    input_img = Input(name="image_input", shape=input_img_shape)
    if image_mode == "simple-shift":
        front = HexConvLayer(filters=32, kernel_size=(3,3), name="encoder_hex_conv_layer")(input_img)
    elif image_mode == "simple":
        front = Conv2D(name="encoder_conv_layer_0",
                       filters=32, kernel_size=(3,3),
                       kernel_initializer="he_uniform",
                       padding = "valid",
                       activation="relu")(input_img)
        front = MaxPooling2D(name=f"encoder_max_poolin_layer_0", pool_size=(2, 2))(front)
    else:
        raise ValueError(f"Invalid image mode {image_mode}")

    ## convolutional layers
    conv_kernel_sizes = conv_kernel_sizes if conv_kernel_sizes is not None else []
    filters = 32
    i = 1
    for kernel_size in conv_kernel_sizes:
        front = Conv2D(name=f"encoder_conv_layer_{i}_a",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_{i}_a", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{i}_a")(front)
        front = Conv2D(name=f"encoder_conv_layer_{i}_b",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_{i}_b", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{i}_b")(front)
        front = MaxPooling2D(name=f"encoder_maxpool_layer_{i}", pool_size=(2,2))(front)
        filters *= 2
        i += 1

    ## generate latent variables by 1x1 Convolutions
    if telescope == "LST_LSTCam":
        kernel_size = (3, 2)
    elif telescope == "MST_FlashCam":
        kernel_size = (5, 1)
    elif telescope == "SST1M_DigiCam":
        kernel_size = (4, 1)
        
    front = Conv2D(name=f"encoder_conv_layer_compress",
                   filters=filters, kernel_size=kernel_size,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu")(front)
    front = Conv2D(name="encoder_conv_layer_to_latent",
                   filters=latent_variables, kernel_size=1,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu")(front)
    front = Flatten(name="encoder_flatten_to_latent")(front)
    
    # Logic Block
    ## extra Telescope Features
    input_params = Input(name="feature_input", shape=input_features_shape)
    front = Concatenate()([input_params, front])

    ## dense blocks
    l2_ = lambda activity_regularizer_l2: None if activity_regularizer_l2 is None else l2(activity_regularizer_l2)
    for dense_i in range(dense_layer_blocks):
        front = Dense(name=f"logic_dense_{dense_i}", units=latent_variables//2, kernel_regularizer=l2_(activity_regularizer_l2))(front)
        front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)

    # Outpout block
    dense_i += 1
    front = Dense(units=128, name=f"logic_dense_{dense_i}")(front)
    front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
    front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
    dense_i += 1
    front = Dense(units=64, name=f"logic_dense_{dense_i}")(front)
    front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
    front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
    output = Dense(len(targets), activation="linear")(front)

    model_name = f"CNN_DET_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    return model

#the class BMO_DET inherits the structure of ModelAssembler class, which is defined in assembler.py
class CNN_DET(ModelAssembler):
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_resolutions=tuple(), target_shapes=(),
                 assembler_mode="mean", point_estimation_mode=None, custom_objects=CUSTOM_OBJECTS):

        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, \
                         lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=CUSTOM_OBJECTS)
        
        if assembler_mode not in ['mean']:
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")

        self.assemble_mode = assembler_mode
        self.point_estimation_mode = point_estimation_mode
        self.target_resolutions = target_resolutions
    
    def model_estimation(self, x_i_telescope, telescope, verbose=0, **kwargs):
        model_telescope = self.models[telescope]
        return model_telescope.predict(x_i_telescope, verbose=verbose, **kwargs)

    def point_estimation(self, y_predictions):
        return y_predictions

    #expected value using the assembling of several telescopes    
    def assemble(self, y_i_by_telescope):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "mean":
            yi_assembled = self.mean(y_i_all)
        return yi_assembled

    def mean(self, y_i):
        return np.mean(y_i, axis=0)