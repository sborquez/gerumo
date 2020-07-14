"""
Dataset Visualizations
======================

Generate plot for display dataset, images and targets.

Here you can find dataset exploration and samples visualizations.
"""
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import norm, multivariate_normal, rv_continuous, gaussian_kde
import numpy as np
import pandas as pd
import ctaplot

__all__ = [
  'show_input_sample', 'show_target_sample',
  'plot_array', 'plot_telescope_geometry'
]

def show_input_sample(input_image_sample, input_image_mode, input_features_sample, make_simple=False, ax=None):
  pass

def show_target_sample(target_sample, targets, target_mode, target_domains, ax=None):
  pass

def plot_array(array_data):
  plt.figure(figsize=(8, 8))
  plt.title("Array info")
  markers = ['x', '+', 'v',  '^', ',','<', '>', 's',',', 'd']
  for i, (type_, telecopes_) in enumerate(array_data.items()):
      points = list(zip(*[(t['x'], t['y'] )for t in telecopes_.values()]))
      plt.scatter(points[0], points[1], label=type_, marker=markers[i])
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend()
  plt.show()

def plot_telescope_geometry(tel_type, num_pixels, pixel_positions):
    plt.figure(figsize=(8,8))
    plt.title(f"{tel_type}\n pixels: {num_pixels}")
    plt.scatter(pixel_positions[0], pixel_positions[1])
    plt.show()