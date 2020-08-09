"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""



"""
Custom Loss
============
"""
from .loss import hellinger_loss, crossentropy_loss, mae_loss, mse_loss, mean_distance_loss, negloglike_loss
LOSS = {
    "hellinger_loss": hellinger_loss,
    "mse_loss": mse_loss,
    "mae_loss": mae_loss,
    "mean_distance_loss": mean_distance_loss,
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
    "loss": crossentropy_loss(), #dummy loss
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
from .pumonna import pumonna_unit, ParametricUmonna
from .bmo import bmo_unit, BMO

MODELS = {
    "umonna_unit": umonna_unit,
    "pumonna_unit": pumonna_unit,
    "bmo_unit": bmo_unit,
}

from .assembler import ModelAssembler
ASSEMBLERS = {
    "umonna": Umonna,
    "pumonna": ParametricUmonna,
    "bmo": BMO,
}



