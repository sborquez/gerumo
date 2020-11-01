"""
Tools
======

Fix, debug or modify models
"""


__all__ = ['split_model']

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def split_model(model, split_layer_name=None, split_layer_index=None, mc_dropout_layer_prefix="bayesian_"):
    """
    Split a trained model for predictions.

    If `split_layer_name` and `split_layer_index` are both provided, `split_layer_index` will take precedence.
    Indices are based on order of horizontal graph traversal (bottom-up).

    Parameters
    ----------
    model : `tensorflow.keras.Model`
        Source trained model.
    split_layer_name : `str` or `None`
        Name of layer to split model. 
    split_layer_index : `int` or `None`
        Telescope type.
    Returns
    =======
        `tuple` of `tensorflow.keras.Model`
        Encoder and Regressor models with source model's weigths.
    """
    # First model
    split_layer_index = split_layer_index or model.layers.index(model.get_layer(split_layer_name))

    encoder = Model(
        model.input,
        model.get_layer(index=split_layer_index).output
    )
    latent_variables_shape = encoder.output.shape[1:]
    # Seconad model
    x = regressor_input = Input(shape=latent_variables_shape)
    for layer in model.layers[split_layer_index + 1:]:
        if mc_dropout_layer_prefix is not None and\
            mc_dropout_layer_prefix in layer.name:
            x = layer(x, training=True)
        else:
            x = layer(x)
    regressor = Model(regressor_input, x)
    ## copy weights
    for layer in regressor.layers[1:]:
        layer.set_weights(
            model.get_layer(name=layer.name).get_weights()
        )
    return encoder, regressor