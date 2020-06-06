"""
parametric UMoNNA
======

Uncertain Multi Observer Neural Network Assembler
with parametric distributions models, Mixture Density Networks
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add,
    Conv2D, MaxPooling2D, Conv3D,
    Conv2DTranspose, Conv3DTranspose,
    UpSampling1D, UpSampling2D, UpSampling3D,
    AveragePooling1D, AveragePooling2D, AveragePooling3D,
    Dense, Flatten, Concatenate, Reshape,
    Activation, BatchNormalization, Dropout
)

from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax

class ParametricUmonna(ModelAssembler):
    def __init__(self):
        pass

    def assemble(self):
        pass
