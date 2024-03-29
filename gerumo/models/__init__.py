"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""
from .tools import *

"""
Custom Loss
============
"""
from .loss import (
    hellinger_loss, crossentropy_loss, mae_loss, mse_loss, 
    negloglike_loss, focal_loss
)
LOSS = {
    "hellinger_loss": hellinger_loss,
    "mse_loss": mse_loss,
    "mae_loss": mae_loss,
    "focal_loss": focal_loss,
    "crossentropy_loss": crossentropy_loss,
    "negloglike_loss": negloglike_loss,
}


"""
Custom Layers
==============
"""
from .layers import HexConvLayer, softmax, MultivariateNormalTriL
LAYERS = {
    "HexConvLayer": HexConvLayer,
}


"""
Custom Objects
==========
"""
CUSTOM_OBJECTS = {
    # LOSSES
    "loss": mae_loss(), #dummy loss
    **LOSS,
    # LAYERS
    **LAYERS
}


"""
Optimizers
==========
"""
from tensorflow.keras.optimizers import (Adam, SGD, RMSprop)
OPTIMIZERS = {
    "adam"    : Adam,
    "sgd"     : SGD,
    "rmsprop" : RMSprop  
}


"""
Models and Assemblers
==========
"""
from .umonna import umonna_unit, Umonna
#from .pumonna import pumonna_unit, ParametricUmonna
from .bmo import bmo_unit, BMO
from .cnn_det import cnn_det_unit, CNN_DET
#from .tiny import tiny_unit, TINY
#from .multiresolution import multiresolution_unit, MultiResolution

MODELS = {
    "umonna_unit": umonna_unit,
    #"pumonna_unit": pumonna_unit,
    "bmo_unit": bmo_unit,
    "cnn_det_unit": cnn_det_unit,
    #"tiny_unit": tiny_unit,
    #"multiresolution_unit": multiresolution_unit
}

from .assembler import ModelAssembler
ASSEMBLERS = {
    "umonna": Umonna,
    #"pumonna": ParametricUmonna,
    "bmo": BMO,
    "cnn_det": CNN_DET,
    #"tiny": TINY,
    #"multiresolution": MultiResolution
}



