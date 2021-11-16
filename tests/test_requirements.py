def test_tensorflow():
    import tensorflow as tf
    assert tf.__version__ == '2.6.0'
    import tensorflow_probability as tfp
    assert tfp.__version__ == '0.14.0'
    import tensorflow_addons as tfa
    assert tfa.__version__ == '0.15.0'
    import tensorboard
    
def test_scipy():
    import pandas
    import sklearn

def test_utils():
    import fvcore
    import tables
    
def test_cta():
    import ctaplot
    assert ctaplot.__version__ == "0.5.6"