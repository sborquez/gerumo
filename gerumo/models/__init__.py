"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""


from .loss import hellinger_loss, crossentropy_loss, mse_loss, mean_distance_loss, negloglike_loss
from .layers import HexConvLayer, softmax, MultivariateNormalTriL

CUSTOM_OBJECTS = {
    # LOSSES
    "hellinger_loss": hellinger_loss,
    "mse_loss": mse_loss,
    "mean_distance_loss": mean_distance_loss,
    "crossentropy_loss": crossentropy_loss,
    "negloglike_loss": negloglike_loss,
    "loss": crossentropy_loss(), #dummy loss
    # LAYERS
    "HexConvLayer": HexConvLayer,
}

from .umonna import umonna_unit, Umonna
from .pumonna import pumonna_unit, ParametricUmonna

MODELS = {
    "umonna_unit": umonna_unit,
    "pumonna_unit": pumonna_unit
}

from .assembler import ModelAssembler
ASSEMBLERS = {
    "umonna": Umonna,
    "pumonna": ParametricUmonna
}

