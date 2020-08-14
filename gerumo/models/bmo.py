"""
BMO: Bayesian Multi Observer
======
Bayesian Neural Network for ,ono and multi-stereo event reconstruction
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
    Activation, BatchNormalization, Dropout,
    ActivityRegularization
)
from tensorflow.keras.regularizers import l2
from . import CUSTOM_OBJECTS
from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax


def bmo_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                latent_variables=800, dense_layer_blocks=8, dropout_rate=0.3,
                activity_regularizer_l2=None):
    """Build BMO Unit Model
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
        front = MaxPooling2D(name=f"encoder_maxpool_layer_0", pool_size=(2, 2))(front)
    else:
        raise ValueError(f"Invalid image mode {image_mode}")

    ## convolutional layers
    conv_kernel_sizes = [5, 3, 3]
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
    
    # Skip Connection
    l2_ = lambda activity_regularizer_l2: None if activity_regularizer_l2 is None else l2(activity_regularizer_l2)
    skip_front = front
    skip_front = Dense(name=f"logic_dense_shortcut", units=latent_variables//2, kernel_regularizer=l2_(activity_regularizer_l2))(skip_front)
    skip_front = Activation(name=f"logic_ReLU_shortcut", activation="relu")(skip_front)
    skip_front = BatchNormalization(name=f"logic_batchnorm_shortcut")(skip_front)
    skip_front = Dropout(name=f"bayesian_Dropout_shortcut", rate=dropout_rate)(skip_front, training=True)

    # Logic Block
    ## extra Telescope Features
    input_params = Input(name="feature_input", shape=input_features_shape)
    front = Concatenate()([input_params, front])

    ## dense blocks
    for dense_i in range(dense_layer_blocks):
        front = Dense(name=f"logic_dense_{dense_i}", units=latent_variables//2, kernel_regularizer=l2_(activity_regularizer_l2))(front)
        front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
        front = Dropout(name=f"bayesian_Dropout_{dense_i}", rate=dropout_rate)(front, training=True)

    # Add Skip connection
    front = Add()([front, skip_front])
    front = Reshape((1, 1,  latent_variables//2), name="logic_reshape")(front)

    front = Conv2D(name=f"logic_dense_last", kernel_size=1, 
                    filters=latent_variables//2,
                    kernel_initializer="he_uniform")(front)
    front = Activation(activation="relu")(front)
    front = front if activity_regularizer_l2 is None else ActivityRegularization(l2=activity_regularizer_l2)(front)
    front = BatchNormalization()(front)
    front = Dropout(name=f"bayesian_Dropout_{dense_i+1}", rate=dropout_rate)(front, training=True)
    
    # Output block
    front = Flatten()(front)
    target_fronts = []
    for target_i in range(len(targets)):
        dense_i_t = dense_i + 1
        target_front = Dense(units=64, kernel_regularizer=l2_(activity_regularizer_l2))(front)
        target_front = Activation(activation="relu")(target_front)
        target_front = BatchNormalization()(target_front)
        target_front = Dropout(name=f"bayesian_Dropout_{dense_i_t+1}_{target_i}", rate=dropout_rate)(target_front, training=True)
        
        target_front = Dense(units=64, kernel_regularizer=l2_(activity_regularizer_l2))(target_front)
        target_front = Activation(activation="relu")(target_front)
        target_front = BatchNormalization()(target_front)
        target_front = Dropout(name=f"bayesian_Dropout_{dense_i_t+2}_{target_i}", rate=dropout_rate)(target_front, training=True)

        target_front = Dense(1)(target_front)
        target_fronts.append(target_front)

    output = Concatenate()(target_fronts)
    model_name = f"BMO_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    return model


class BMO(ModelAssembler):
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_resolutions=tuple(), target_shapes=(),
                 assembler_mode="resample", point_estimation_mode="expected_value", custom_objects=CUSTOM_OBJECTS):

        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=CUSTOM_OBJECTS)
        if assembler_mode not in ['resample', 'normalized_product']:
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")
        if point_estimation_mode not in ["expected_value"]:
            raise ValueError(f"Invalid point_estimation_mode: {point_estimation_mode}")

        self.assemble_mode = assembler_mode
        self.point_estimation_mode = point_estimation_mode
        self.target_resolutions = target_resolutions
        self.sample_size = 200

    @staticmethod
    def bayesian_estimation(model, x_i_telescope, sample_size, verbose, **kwargs):
        y_predictions_points = np.array([model.predict(x_i_telescope, verbose=verbose, **kwargs) for _ in range(sample_size)])
        y_predictions_points = np.swapaxes(np.swapaxes(y_predictions_points, 0, 1), 1, 2)
        y_predictions_kde    = [st.gaussian_kde(y_predictions_point) for y_predictions_point in y_predictions_points]
        return y_predictions_kde, y_predictions_points
    
    def model_estimation(self, x_i_telescope, telescope, verbose, **kwargs):
        model_telescope = self.models[telescope]
        y_predictions, _ = BMO.bayesian_estimation(model_telescope, x_i_telescope, self.sample_size, verbose, **kwargs)
        return y_predictions

    def point_estimation(self, y_predictions):
        if self.point_estimation_mode == "expected_value":
            y_point_estimations = self.expected_value(y_predictions)
        return y_point_estimations

    def expected_value(self, y_predictions):
        if isinstance(y_predictions[0], st.gaussian_kde):
            y_mus = np.array([y_i.resample(int(2.5e6)).mean(axis=1) for y_i in y_predictions])
            return y_mus

    def assemble(self, y_i_by_telescope):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "resample":
            yi_assembled = self.resample(y_i_all)
        elif self.assemble_mode == "normalized_product":
            yi_assembled = self.normalized_product(y_i_all)
        return yi_assembled

    def resample(self, y_i):
        resamples = np.array(np.hstack([y_kde_j.resample(int(1e5)) for y_kde_j in y_i]))
        return st.gaussian_kde(resamples)

    def normalized_product(self, y_i):
        raise NotImplementedError
