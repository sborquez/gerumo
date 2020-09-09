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
from . import CUSTOM_OBJECTS
from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax


def cnn_det_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                latent_variables=200, dense_layer_blocks=5, dropout_rate=0.3):
    """Build CNN Unit Model
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
    conv_kernel_sizes = [5, 3, 3]
    filter_sizes = [32, 64, 128]

    con = 0
    for con in range(3):

        kernel_size = conv_kernel_sizes[con]
        filters = filter_sizes[con]
        
        front = Conv2D(name=f"encoder_conv_layer_{2*con+1}_a",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{2*con+1}_a")(front)
        front = Activation(name=f"encoder_ReLU_{2*con+1}_a", activation="relu")(front)

        front = Conv2D(name=f"encoder_conv_layer_{2*con+2}_b",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)        
        front = BatchNormalization(name=f"encoder_batchnorm_{2*con+2}_b")(front)
        front = Activation(name=f"encoder_ReLU_{2*con+2}_b", activation="relu")(front)
        
        front = MaxPooling2D(name=f"encoder_maxpool_layer_{2*con+1}", pool_size=(2,2))(front)

    ## generate latent variables by 1x1 Convolutions
    if telescope == "LST_LSTCam":
        kernel_size = (3, 2)
    elif telescope == "MST_FlashCam":
        kernel_size = (5, 1)
    elif telescope == "SST1M_DigiCam":
        kernel_size = (4, 1)

    #twice the last filter
    filters = 256
    front = Conv2D(name=f"encoder_conv_layer_compress",
                   filters=filters, kernel_size=kernel_size,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu")(front)

    latent_variables=512
    front = Conv2D(name="encoder_conv_layer_to_latent",
                   filters=latent_variables, kernel_size=1,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu")(front)
    
    front = Flatten(name="encoder_flatten_to_latent")(front)

    ## extra Telescope Features
    input_params = Input(name="feature_input", shape=input_features_shape)
    front = Concatenate()([input_params, front])

    ## dense blocks
    dense_layer_blocks = 3
    for dense_i in range(dense_layer_blocks):

        units = latent_variables//(2**(dense_i+1))
        
        front = Dense(name=f"logic_dense_{dense_i}", units=units)(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
        front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
        
    output = Dense(len(targets), activation="linear")(front)

    model_name = f"CNN_DET2_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    model.summary()
    return model

#the class BMO_DET inherits the structure of ModelAssembler class, which is defined in assembler.py
class CNN_DET(ModelAssembler):
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_resolutions=tuple(), target_shapes=(),
                 assembler_mode="resample", point_estimation_mode="expected_value", custom_objects=CUSTOM_OBJECTS):

        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, \
                         lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=CUSTOM_OBJECTS)
        
        if assembler_mode not in ['resample', 'normalized_product']:
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")
        if point_estimation_mode not in ["expected_value"]:
            raise ValueError(f"Invalid point_estimation_mode: {point_estimation_mode}")

        self.assemble_mode = assembler_mode
        self.point_estimation_mode = point_estimation_mode
        self.target_resolutions = target_resolutions
        self.sample_size = 1
        
    @staticmethod
    def det_estimation(model, x_i_telescope, verbose, **kwargs):
        y_predictions_points = model.predict(x_i_telescope, verbose=verbose, **kwargs)
        #print(y_predictions_points.shape)
        return y_predictions_points
    
    def model_estimation(self, x_i_telescope, telescope, verbose, **kwargs):
        model_telescope = self.models[telescope]
        y_predictions = CNN_DET.det_estimation(model_telescope, x_i_telescope, verbose, **kwargs)
        return y_predictions

    #expected value in the space of Bayesian predictions
    def point_estimation(self, y_predictions):
        if self.point_estimation_mode == "expected_value":
            y_point_estimations = self.expected_value(y_predictions)
        return y_point_estimations

    def expected_value(self, y_predictions):
        if isinstance(y_predictions[0], st.gaussian_kde):
            y_mus = np.array([y_i.resample(int(100)).mean(axis=1) for y_i in y_predictions])
            return y_mus

    #expected value using the assembling of several telescopes    
    def assemble(self, y_i_by_telescope):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "resample":
            yi_assembled = self.resample(y_i_all)
        elif self.assemble_mode == "normalized_product":
            yi_assembled = self.normalized_product(y_i_all)
        return yi_assembled

    #kde in the telescope space, later we compute the mean over mc repetitions
    #since we cannot save the instance of the mc repetitions we resample using the obtained kdes
    def resample(self, y_i):
        resamples = np.array(np.hstack([y_kde_j.resample(int(100)) for y_kde_j in y_i]))
        return st.gaussian_kde(resamples)

    def normalized_product(self, y_i):
        raise NotImplementedError
 
