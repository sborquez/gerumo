import sys
sys.path.insert(1, '..')

from gerumo import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import timeline

from time import time
import uuid
from os.path import abspath, basename, splitext
import argparse

import logging


def test_gpu():
    cuda = tf.test.is_built_with_cuda()
    if cuda:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            #logging.info(f"Availables GPUs: {len(gpus)}" )
            print(f"Availables GPUs: {len(gpus)}" )
            for i, gpu in enumerate(gpus):
                #logging.info(f"Availables GPU {i}: {gpu}" )
                print(f"Availables GPU {i}: {gpu}" )
        else:
            #logging.info("Not availables GPUs.")
            print("Not availables GPUs.")
    else:
        #logging.info("Tensorflow is not built with CUDA")
        print("Tensorflow is not built with CUDA")

def bottle_neck(model_path):
    raise NotImplementedError

def save_plot_model(model_path):
    raise NotImplementedError

ap = argparse.ArgumentParser(description="Profiling and debuging tools to find bottlenecks in the architecture and other tests.")
ap.add_argument("--gpu", dest="gpu", action="store_true")

args = vars(ap.parse_args())
gpu = args["gpu"]

if gpu:
    print("Running GPU tests.")
    test_gpu()
print("Done")