import tensorflow as tf
from tf import keras
import tensorflow.keras.backend as K
import numpy as np
from functools import reduce

def crossentropy_loss(dimensions=3, epsilon=1e-16):
    """Return the Cross Entropy loss function for multidimension probability map."""
    def loss(y_true, y_pred):
        """Cross entropy loss function."""
        axis = [-i for i in range(1, dimensions+1)]
        return K.sum(-K.log(y_pred + epsilon)*y_true, axis=axis)
    return loss

def hellinger_loss():
    """Return the Hellinger loss function for multidimension probability map."""
    sqrt_two = 1.0/np.sqrt(2)
    def loss(y_true, y_pred):
        """Hellinger loss or Helliger distance,  is distance between two discrete distributions.
        
        Hellinger distance:
            H(p,q)= 1/sqrt(2) * sqrt(Sum((sqrt(p_i) - sqrt(q_i))**2)) 
        """
        # Reshape tensors (batch_flatten)
        y_true_ = K.batch_flatten(y_true)
        y_pred_ = K.batch_flatten(y_pred)
        # Hellinger distance
        z = K.sqrt(y_true_) - K.sqrt(y_pred_)
        return sqrt_two * K.sqrt(K.batch_dot(z, z))
    return loss

def mse_loss():
    """Return the Mean Square loss function for multidimension probability map."""
    return keras.losses.mean_squared_error

def mean_distance_loss(shape):
    """Return the Mean Distances Weighted loss function for multidimension distance map."""
    c_inv = 1/reduce(lambda a,b: a*b, shape)
    def loss(y_true, y_pred):
        """Mean Distances Weighted loss function

        This use a predicted probability map and a distance map from the target and weight 
        its distances by the predcted probability.
        
        MDW(Dp, q) : 1/c Sum((Dp_i*q_i)**2)
        """
        # Reshape tensors (batch_flatten)
        y_true_ = K.batch_flatten(y_true)
        y_pred_ = K.batch_flatten(y_pred)
        return c_inv * K.sum(K.pow(y_true_ * y_pred_, 2), axis=-1)
    return loss

def wasserstein_loss():
    """Return the Wasserstein loss function for multidimension probability map."""
    raise NotImplementedError
    def loss(y_true, y_pred):
        """Wasserstein loss function."""
        return 0
    return loss