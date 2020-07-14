"""
Dataset Visualizations
======================

Generate plot for display dataset, images and targets.

Here you can find dataset exploration and samples visualizations.
"""
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_telescope_geometry(tel_type, pixel_positions, num_pixels=None):
    shape = pixel_positions.shape
    if num_pixels is None:
        num_pixels = shape[-1]
    colors = cm.rainbow(np.linspace(0, 1, num_pixels))
    if shape[0] == 2:
        plt.figure(figsize=(8,8))
        plt.title(f"{tel_type}\n pixels: {num_pixels}")
        plt.scatter(pixel_positions[0], pixel_positions[1], color=colors)
    elif shape[0] == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        plt.suptitle(f"{tel_type}\n pixels: {num_pixels}")        
        ax1.scatter(pixel_positions[0], pixel_positions[2], color=colors)
        ax2.scatter(pixel_positions[1], pixel_positions[2], color=colors)
    plt.show()

def plot_dataset_targets(dataset, targets=["alt", "az", "log_mc_energy"]):
    pass

def plot_observation_scatter(charge, peakpos, pixel_positions, telescope_type=None, event_unique_id=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1.set_title("Charge")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax1.scatter(pixel_positions[0], pixel_positions[1], c=charge)
    plt.colorbar(im, cax=cax)

    plt.colorbar(im, cax=cax)
    ax2.set_title("Peak Pos")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax2.scatter(pixel_positions[0], pixel_positions[1], c=peakpos)
    plt.colorbar(im, cax=cax)

    if event_unique_id is None:
        title = f"{'' if telescope_type is None else telescope_type}"
    else:
        title = f"{'' if telescope_type is None else telescope_type}\nevent: {event_unique_id}"

    plt.suptitle(title);
    plt.show()

def plot_observation_input():
    pass