"""
Custom Layers
=============

Collection of models custom layers.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Lambda, Concatenate, MaxPooling2D, Reshape


"""
Adapted from https://gist.github.com/raingo/a5808fe356b8da031837

Multi dimensional softmax,
refer to https://github.com/tensorflow/tensorflow/issues/210
compute softmax along the dimension of target
the native softmax only supports batch_size x dimension
"""
def softmax(target, axis, name=None):
    max_axis = tf.reduce_max(target, axis, keepdims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
    softmax = target_exp / normalize
    return softmax


class HexConvLayer(Layer):
    def __init__(self, filters, kernel_size=(3,3), name=None, reshape=None, **kargs):
        super(HexConvLayer, self).__init__(name=name, **kargs)
        self.__name = name
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2d = Conv2D(filters, kernel_size, kernel_initializer="he_uniform",
                             padding = "valid", activation="relu", strides=(2,1))
        self.left_split = Lambda(lambda x: x[:,0,:-1,:,:])
        self.right_split = Lambda(lambda x: x[:,1,1:,:,:])
        self.concatenate = Concatenate(axis=2)
        self.pooling = MaxPooling2D(pool_size=(2,2))
        self.__reshape = reshape
        if self.__reshape is not None:
            self.reshape = Reshape(self.__reshape)
        # Infer shape in running time
        else:
            self.reshape = None
    
    #@tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        input_left = self.left_split(inputs)
        input_right = self.right_split(inputs)

        front_left = self.conv2d(input_left)
        front_right = self.conv2d(input_right)
        
        front = self.concatenate([front_left, front_right])
        
        # Intercalate on axis Y the application of the convolution.
        cshape = front.get_shape()
        if self.reshape is None:
            self.__reshape = (int(cshape[1])*2,int(cshape[2])//2,int(cshape[3]))
            self.reshape = Reshape(self.__reshape)
        front = self.reshape(front)

        return self.pooling(front)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "name": self.__name,
            "reshape": self.__reshape
        })
        return config
        
