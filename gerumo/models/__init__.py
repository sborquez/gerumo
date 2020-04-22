"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""


from .loss import hellinger_loss, crossentropy_loss, mse_loss, mean_distance_loss
from .layers import HexConvLayer, softmax
from .base import AssemblerModel
from .umonna import umonna_unit

CUSTOM_OBJECTS = {
    # LOSSES
    "hellinger_loss": hellinger_loss,
    "mse_loss": mse_loss,
    "mean_distance_loss": mean_distance_loss,
    "crossentropy_loss": crossentropy_loss,
    # LAYERS
    "HexConvLayer": HexConvLayer,
}

MODELS = {
    "umonna_unit": umonna_unit,
}