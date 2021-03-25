"""
Loss Functions
==============

Collections of different Loss function for probability distributions
and distance matrix.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from functools import reduce

def crossentropy_loss(dimensions=3):
    """Return the Cross Entropy loss function for multidimension probability map."""
    axis = [-i for i in range(1, dimensions+1)]
    epsilon = tf.keras.backend.epsilon()
    def loss(y_true, y_pred):
        """Cross entropy loss function."""
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis)
    return loss

def focal_loss(dimensions=3, alphas=None, gamma=2.0):
    """Return the focal loss function for multidimension probability map."""
    axis = [-i for i in range(1, dimensions+1)]
    epsilon = tf.keras.backend.epsilon()
    alphas = alphas or 1
    def loss(y_true, y_pred):
        """Focal loss function."""
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred) * tf.math.pow(1 - y_pred, gamma) * alphas, axis)
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

def mae_loss():
    """Return the Mean Absolute loss function for multidimension probability map."""
    return keras.losses.mae

def negloglike_loss(dimensions):
    def loss(y_true, y_params_pred):
        return -y_params_pred.log_prob(y_true)*tf.linalg.norm(y_params_pred.mean() - y_true, axis=-1) 
    return loss

def mve_loss(dimensions):
    def loss(y_true, y_params_pred):
        return -y_params_pred.log_prob(y_true)*tf.linalg.norm(y_params_pred.loc - y_true, axis=-1)  + tf.linalg.trace(y_params_pred.covariance()) 
    return loss