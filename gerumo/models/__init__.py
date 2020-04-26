"""
Models
======

Custom models, meta-models, layers and loss funtions.

"""


from .loss import hellinger_loss, crossentropy_loss, mse_loss, mean_distance_loss
from .layers import HexConvLayer, softmax
CUSTOM_OBJECTS = {
    # LOSSES
    "hellinger_loss": hellinger_loss,
    "mse_loss": mse_loss,
    "mean_distance_loss": mean_distance_loss,
    "crossentropy_loss": crossentropy_loss,
    # LAYERS
    "HexConvLayer": HexConvLayer,
}

from .umonna import umonna_unit, Umonna
MODELS = {
    "umonna_unit": umonna_unit,
}

from .assembler import ModelAssembler
ASSEMBLERS = {
    "umonna": Umonna,
}

