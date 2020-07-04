"""
parametric UMoNNA
======

Uncertain Multi Observer Neural Network Assembler 
with parametric distributions models
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


def pumonna_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                latent_variables=800, dense_layer_blocks=5):
    """Build Pumonna Unit Model
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
        front = MaxPooling2D(name=f"encoder_conv_layer_0", pool_size=(2, 2))(front)
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
        front = Activation(name=f"encoder_ReLU_layer_{i}_a", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{i}_a")(front)
        front = Conv2D(name=f"encoder_conv_layer_{i}_b",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_layer_{i}_b", activation="relu")(front)
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
    skip_front = front
    skip_front = Dense(name=f"logic_dense_shortcut", units=latent_variables//2)(skip_front)
    skip_front = Activation(name=f"logic_ReLU_layer_shortcut", activation="relu")(skip_front)
    skip_front = BatchNormalization(name=f"logic_batchnorm_shortcut")(skip_front)

    # Logic Block
    ## extra Telescope Features
    input_params = Input(name="feature_input", shape=input_features_shape)
    front = Concatenate()([input_params, front])

    ## dense blocks
    for dense_i in range(dense_layer_blocks):
        front = Dense(name=f"logic_dense_{dense_i}", units=latent_variables//2)(front)
        front = Activation(name=f"logic_ReLU_layer_{dense_i}", activation="relu")(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
        front = Dropout(name=f"logic_Dropout_layer_{dense_i}", rate=0.25)(front)

    # Add Skip connection
    front = Add()([front, skip_front])
    front = Reshape((1, 1,  latent_variables//2), name="logic_reshape")(front)

    front = Conv2D(name=f"logic_dense_last", kernel_size=1, 
                   filters=latent_variables//2,
                   kernel_initializer="he_uniform")(front)
    front = Activation(activation="relu")(front)
    front = BatchNormalization()(front)

    if target_mode == "lineal":
        front = Dense(units=64)(front)
        front = Activation(activation="tanh")(front)
        front = BatchNormalization()(front)
        front = Dense(units=64)(front)
        front = Activation(activation="tanh")(front)
        front = BatchNormalization()(front)
        front = Dense(tfp.layers.MultivariateNormalTriL.params_size(len(targets)), activation=None)(front)
        output = tfp.layers.MultivariateNormalTriL(len(targets), name='pdf')(front)
    else:
        raise ValueError(f"Invalid target_mode: '{target_mode}'" )

    model_name = f"Pumonna_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    return model


class ParametricUmonna(ModelAssembler):
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_shapes=(),
                 assembler_mode="normalized_product", point_estimation_mode="expected_value", custom_objects=CUSTOM_OBJECTS):
        

        # add tensorflow probability layers
        copy_custom_objects = dict(custom_objects)
        copy_custom_objects['MultivariateNormalTriL'] = ParametricUmonna.MultivariateNormalTriL_loader(len(targets))
        
        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=copy_custom_objects)
        if assembler_mode not in ['normalized_product', "gaussian_barycenter_W2"]:
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")
        self.assemble_mode = assembler_mode
        
        if point_estimation_mode not in ["expected_value"]:
            raise ValueError(f"Invalid point_estimation_mode: {point_estimation_mode}")
        self.point_estimation_mode = point_estimation_mode
        self.barycenter_fixpoint_iterations = 25
        self.target_resolutions = None

    @staticmethod
    def MultivariateNormalTriL_loader(event_size):
        """
        helper function to loading model from checkpoint 
        """
        def load_MultivariateNormalTriL(name, trainable, dtype, function, function_type, module, output_shape, output_shape_type, output_shape_module, arguments, make_distribution_fn, convert_to_tensor_fn):
            return tfp.layers.MultivariateNormalTriL(event_size, name=name, trainable=trainable, dtype=dtype, convert_to_tensor_fn=convert_to_tensor_fn)
        return load_MultivariateNormalTriL
    
    def model_estimation(self, x_i_telescope, telescope, verbose, **kwargs):
        model_telescope = self.models[telescope]
        y_predictions_batch = model_telescope(x_i_telescope)
        
        #split batch in individuals distributions
        locs = y_predictions_batch.loc.numpy().squeeze(axis=(1,2))
        scales = y_predictions_batch.scale.to_dense().numpy().squeeze(axis=(1,2))
        y_predictions = [tfp.distributions.MultivariateNormalTriL(loc, scale) for loc, scale in zip(locs, scales)]
        return y_predictions

    def point_estimation(self, y_predictions):
        if self.point_estimation_mode == "expected_value":
            y_point_estimations = self.expected_value(y_predictions)
        return y_point_estimations

    def expected_value(self, y_predictions):
        if isinstance(y_predictions[0], st.rv_continuous):
            y_mus = np.array([y_i.mean() for y_i in y_predictions])
            return y_mus
        else:
            y_mus = np.array([y_i.mean() for y_i in y_predictions])
            return y_mus

    def assemble(self, y_i_by_telescope):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "gaussian_barycenter_W2":
            yi_assembled = self.gaussian_barycenter_W2(y_i_all)
        elif self.assemble_mode == "normalized_product":
            yi_assembled = self.normalized_product(y_i_all)
        return yi_assembled

    def normalized_product(self, y_i):
        return normalized_product_gen(y_i, self.target_domains)

    def gaussian_barycenter_W2(self, y_i):
        raise NotImplementedError
        return None
        # Parameters
        n = len(y_i)
        mus =    y_i[:, 0]
        sigmas = y_i[:, 1:]
        dim = len(self.targets)
        
        # Sigma
        barycenter_sigma = np.eye(dim,dim)
        sqrtm = np.vectorize(sp.linalg.sqrtm, signature="(n,n)->(n,n)")
        for _ in range(barycenter_fixpoint_iterations):
            barycenter_sigma_sqrt = sp.linalg.sqrtm(barycenter_sigma)
            barycenter_sigma  = sqrtm(barycenter_sigma_sqrt@sigmas@barycenter_sigma_sqrt).mean(axis=0)
        # Mu
        barycenter_mu = mus.mean(axis=0, keepdims=True)
        compressed = np.vstack((barycenter_mu, barycenter_sigma))
        return compressed

class normalized_product_gen(st.rv_continuous):
    "Product of Gaussian distributions"
    def __init__(self, distributions, integration_domain=[[-5,5],[-5,5]], **kwargs):
        super().__init__(**kwargs)    
        self.distributions = distributions
        self.integration_domain = integration_domain
        self.dim = distributions[0].event_shape[0]
        if self.dim == 1:
            xmin, xmax = integration_domain[0]
            self.domain = np.arange(xmin, xmax, .05)
        elif self.dim == 2:
            xmin, xmax = integration_domain[0]
            ymin, ymax = integration_domain[1]
            self.domain = np.mgrid[xmin:xmax:(xmax-xmin)/10., ymin:ymax:(ymax-ymin)/10.]
            #self.norm_const = np.sum(self._pdf(np.dstack((self.domain[0], self.domain[1]))))
        elif self.dim == 3:
            raise NotImplementedError

    def _pdf(self, x):
        return np.prod([Ni.prob(x) for Ni in self.distributions], axis=0)
    
    def prob(self, x):
        return self._pdf(x)
    
    def mean(self):
        if self.dim == 2:
            pos = np.dstack((self.domain[0], self.domain[1]))
        else:
            raise NotImplementedError
        prob = self.prob(pos)
        return pos[np.unravel_index(np.argmax(prob, axis=None), prob.shape)]