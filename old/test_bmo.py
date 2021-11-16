import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')


from gerumo import *
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import scipy.stats as st
import timeit
# configuration
config={
"telescope" : "MST_FlashCam",
"image_mode": 'simple-shift',
"image_mask":  True,
"input_img_shape": INPUT_SHAPE["simple-shift-mask"]["MST_FlashCam"],
"input_features_shape": (2,),
"target_mode": "lineal",
"targets": ["alt", "az"]
}
# architecture 
model_extra_params =  {
"latent_variables": 128,
"conv_kernel_sizes": None,
"compress_filters": 256,
"compress_kernel_size": 3,
"dense_layer_units": [128, 128, 64],
"activity_regularizer_l1": None,
"kernel_regularizer_l2": None,
"dropout_rate": 0.5
}
sample_size = 50
batch_size = 32
number_of_runs=2

# model and Ensembler
model = bmo_unit(**config, **model_extra_params)
bmo = BMO(mst_model_or_path=model)
bmo.sample_size=sample_size

# fake data
batch_x = [
    np.float32(np.random.random((batch_size, *INPUT_SHAPE["simple-shift-mask"]["MST_FlashCam"]))),
    np.float32(np.random.random((batch_size, 2)))
]

predict_only = timeit.timeit(lambda: model.predict(batch_x), number=number_of_runs)
print(f"predict one sample: {predict_only/2.0} [s]")

old = timeit.timeit(lambda: bmo.bayesian_estimation_old(model, batch_x, bmo.sample_size, 0), number=number_of_runs)
print(f"mc-dropout vanilla (sample size={sample_size}): {old/2.0} [s]")

new_ = timeit.timeit(lambda: bmo.bayesian_estimation(model, batch_x, bmo.sample_size, 0), number=number_of_runs)
print(f"mc-dropout new (sample size={sample_size}): {new_/2.0} [s]")
new_2nd = timeit.timeit(lambda: bmo.bayesian_estimation(model, batch_x, bmo.sample_size, 0), number=number_of_runs)
print(f"mc-dropout new (sample size={sample_size}) 2nd run: {new_2nd/2.0} [s]")

old_10x = timeit.timeit(lambda: bmo.bayesian_estimation_old(model, batch_x, 10*bmo.sample_size, 0), number=number_of_runs//2)
print(f"mc-dropout vanilla (sample size={10*sample_size}): {old_10x} [s]")

new_10x = timeit.timeit(lambda: bmo.bayesian_estimation(model, batch_x, 10*sample_size, 0), number=number_of_runs)
print(f"mc-dropout new (sample size={10*sample_size}): {new_10x/2.0} [s]")

new_100x = timeit.timeit(lambda: bmo.bayesian_estimation(model, batch_x, 100*sample_size, 0), number=number_of_runs)
print(f"mc-dropout new (sample size={100*sample_size}): {new_100x/2.0} [s]")

