"""
Exploration Visualizations
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
  'plot_input_sample', 'plot_target_sample',
  'plot_array', 'plot_telescope_geometry', 'plot_observation_scatter'
]


"""
Input Samples
================
"""

def show_image_simple(input_image_sample, channels_names=["Charge", "Peak Pos", "Mask"], axis=None):
    # Create new figure
    if axis is None:
        channels = input_image_sample.shape[-1]
        fig, axis = plt.subplots(nrows=1, ncols=channels, figsize=(4*channels,6))

    # for each channel
    for i, ax in enumerate(axis):
        ax.set_title(channels_names if isinstance(channels_names, str) else channels_names[i])
        im = ax.imshow(input_image_sample[:,:,i])
        plt.colorbar(im, ax=ax)
        ax.set_yticks([]); ax.set_xticks([])

    return axis

def show_image_simple_shift(input_image_sample, channels_names=["Charge", "Peak Pos", "Mask"], axis=None):
    # Create new figure
    if axis is None:
        channels = input_image_sample.shape[-1]
        fig, axis = plt.subplots(nrows=2, ncols=channels, figsize=(4*channels, 14))
    for i, shift in ((0,"Left"), (1,"Right")):
        for j, ax in enumerate(axis[i]):
            ax.set_title(f"{channels_names if isinstance(channels_names, str) else channels_names[i]} - {shift}")
            im = ax.imshow(input_image_sample[i, :,:,j], vmin=0)
            #plt.colorbar(im, ax=ax)
            ax.set_yticks([]); ax.set_xticks([])
    return axis

def show_image_raw(input_image_sample, axis=None):
    # Create new figure
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        #TODO: add scatter plot
        raise NotImplementedError
    return axis

def plot_input_sample(input_image_sample, input_image_mode, input_features_sample, title=None, make_simple=False, save_to=None):
    
    if input_image_mode == "simple":
        channels = input_image_sample.shape[-1]
        fig, axis = plt.subplots(nrows=1, ncols=channels, figsize=(4*channels,6))
        show_image_simple(input_image_sample, axis=axis)
    elif input_image_mode == "simple-shift":
        if make_simple:
            channels = input_image_sample.shape[-1]
            fig, axis = plt.subplots(nrows=1, ncols=channels, figsize=(4*channels, 6))
            show_image_simple(input_image_sample[0], axis=axis)
        else:
            channels = input_image_sample.shape[-1]
            fig, axis = plt.subplots(nrows=2, ncols=channels, figsize=(4*channels, 14))
            axis = show_image_simple_shift(input_image_sample, axis=axis)
    elif input_image_mode == "time":
        channels = input_image_sample.shape[-1]
        fig, axis = plt.subplots(nrows=1, ncols=channels, figsize=(4*channels,6))
        show_image_simple(input_image_sample,channels_names=["Peak Pos", 'Mask'], axis=axis)
    elif input_image_mode == "time-shift":
        if make_simple:
            channels = input_image_sample.shape[-1]
            fig, axis = plt.subplots(nrows=1, ncols=channels, figsize=(4*channels, 6))
            show_image_simple(input_image_sample[0], channels_names=["Peak Pos", 'Mask'], axis=axis)
        else:
            channels = input_image_sample.shape[-1]
            fig, axis = plt.subplots(nrows=2, ncols=channels, figsize=(4*channels, 14))
            show_image_simple_shift(input_image_sample, channels_names=["Peak Pos", 'Mask'], axis=axis)
    elif input_image_mode == "raw":
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        show_image_raw(input_image_sample, axis=axis)
    else:
        raise ValueError(f"invalid 'input_image_mode': {input_image_mode}")
    
    if isinstance(title, str):
        title = f"Prediction for event {title}\nTelescope Features: {input_features_sample}"
    elif isinstance(title, tuple):
        title = f"Prediction for event {title[0]}\ntelescope id {title[1]}\nTelescope Features: {input_features_sample}"
    else:
        title = f"Prediction for a event"
    fig.suptitle(title)
    
    # Save or Show
    if save_to is not None:
        fig.savefig(save_to)
        plt.close(fig)
    else:
        plt.show()

"""
Target Samples
================
"""

def plot_target_sample(target_sample, targets, target_mode, target_domains):
    pass


"""
Dataset Visualizations
================
"""

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


def plot_observation_scatter(charge, peakpos, pixel_positions, telescope_type=None, event_unique_id=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1.set_title("Charge")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axis("right", size="5%", pad=0.05)
    im = ax1.scatter(pixel_positions[0], pixel_positions[1], c=charge)
    plt.colorbar(im, cax=cax)

    ax2.set_title("Peak Pos")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axis("right", size="5%", pad=0.05)
    im = ax2.scatter(pixel_positions[0], pixel_positions[1], c=peakpos)
    plt.colorbar(im, cax=cax)

    if event_unique_id is None:
        title = f"{'' if telescope_type is None else telescope_type}"
    else:
        title = f"{'' if telescope_type is None else telescope_type}\nevent: {event_unique_id}"

    plt.suptitle(title);
    plt.show()
