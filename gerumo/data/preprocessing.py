"""
Data and image transformations and normalization
================================================

This module handle telescope data, event data and camera data to normalize
and apply different preprocessing function. This modele also can generate 
different square images from charge and peakpos value.
"""

import numpy as np
from scipy.stats import multivariate_normal
from os import path
from . import IMAGES_SIZE, INPUT_SHAPE, TELESCOPES, PIXELS_POSITION

__all__ = [
    'aggregate_dataset', 'filter_dataset',
    'camera_to_image', 'cameras_to_images',
    'targets_to_matrix'
]


"""
Utils Functions
===============
"""

def aggregate_dataset(dataset, az=True, log10_mc_energy=True, hdf5_file=True):
    """
    Perform simple aggegation to targe columns.

    Parameters
    ==========
    az : `bool`, optional
        Translate domain from [0, 2\pi] to [-\pi, \pi]. (default=False)
    log10_mc_energy : `bool`, optional
        Add new log10_mc_energy column, with the logarithm values of mc_energy.
    Returns
    =======
    `pd.DataFrame`
        Dataset with aggregate information.
    """
    if az:
        dataset["az"] = dataset["az"].apply(lambda rad: np.arctan2(np.sin(rad), np.cos(rad)))
    if log10_mc_energy:
        dataset["log10_mc_energy"] = dataset["mc_energy"].apply(lambda energy: np.log10(energy))
    if hdf5_file:
        dataset["hdf5_filepath"] = dataset[["folder", "source"]].apply(lambda x: path.join(x[0], x[1]), axis=1)
    return dataset

def filter_dataset(dataset, telescopes=[], number_of_observations=[], domain={}):
    """
    Select a subset from the dataset given some restrictions.

    The dataset can be filtered by telescope types, number of observations,
    and by a range of values for the targets. 

    Parameters
    ==========
    telescopes : `list` of `str` or 'str'
        Selected telescopes type for the dataset. 'str' if is just one.
    number_of_observations : `list` of `int` or 'int
        For each telescope in 'telescopes' parameter, the minimum amount
        of observations. 'int' if is just one.
    domain : `dict` [ `str`,  `tuple` of `int`]
        A dictionary with names of columns and their value range.
    Returns
    =======
     `pd.DataFrame`
        Filtered dataset.
    """
    if isinstance(telescopes, str):
        telescopes = [telescopes]
    if isinstance(number_of_observations, int):
        number_of_observations = [number_of_observations]

    # filter telescopes
    filtered_dataset = dataset[dataset.type.isin(telescopes)]
    # # filter number of observatoins
    # FIXME: 
    # for telescope, observations in zip(telescopes, number_of_observations):
    #     filtered_events = filtered_dataset[filtered_dataset.type == telescope]\
    #                         .groupby("event_unique_id")\
    #                         .filter(lambda g: len(g) >= observations)\
    #                         .event_unique_id.unique()
    #     filtered_dataset = filtered_dataset[filtered_dataset.event_unique_id.isin(filtered_events)]
    # filter domain
    selection = np.ones((len(filtered_dataset),), dtype=bool)
    for target, (vmin, vmax) in domain.items():
        selection &= ((vmin <= filtered_dataset[target]) & (filtered_dataset[target] <= vmax))
    return filtered_dataset[selection]


"""
Input preprocessing Functions
=============================

Prepare input images, normalize and standarize values
from images and features. 
"""

def _simple(charge, peakpos, telescope_type, mask):
    x, y = PIXELS_POSITION["simple"][telescope_type] #(x, y)
    image_size = IMAGES_SIZE[telescope_type]
    if mask:
        input_shape = INPUT_SHAPE["simple-mask"][telescope_type]
    else:
        input_shape = INPUT_SHAPE["simple"][telescope_type]
    canvas = np.zeros(input_shape, dtype="float32")
    canvas[y, x, 0] = charge
    canvas[y, x, 1] = peakpos
    if mask:
        canvas[y, x, 2] = 1
    return canvas

def _simple_shift(charge, peakpos, telescope_type, mask):
    x_left, x_right, y = PIXELS_POSITION["simple_shift"][telescope_type] #(x_l, x_r, y)
    image_size = IMAGES_SIZE[telescope_type]
    if mask:
        input_shape = INPUT_SHAPE["simple-shift-mask"][telescope_type]
    else:
        input_shape = INPUT_SHAPE["simple-shift"][telescope_type]
    canvas = np.zeros(input_shape, dtype="float32")
    canvas[0, y, x_left, 0] = charge
    canvas[0, y, x_left, 1] = peakpos
    canvas[1, y, x_right, 0] = charge
    canvas[1, y, x_right, 1] = peakpos
    if mask:
        canvas[0, y, x_left, 2] = 1
        canvas[1, y, x_right, 2] = 1
    return canvas

def _time_split(charge, peakpos, telescope_type, mask):
    raise NotImplementedError

def _time_split_shift(charge, peakpos, telescope_type, mask):
    raise NotImplementedError

def camera_to_image(charge, peakpos, telescope_type, mode="simple", mask=True):
    """
    Transform the charge and peakpos values into an rectangle image.

    The transformation can be done in different modes:
    - 'simple' just return an image two channels (charge and peakpos), 
    where each pixel is vertical aligned. 
    - 'simple-shift' create two version of each image, where odd rows are 
    shifted to right and left, respectively.
    - 'time-split' mode generates just one multichannel image, where the first channel
    is the charge, and the other channels have binary values and represent each value 
    in peakpos, like a one-hot representation. 
    - 'time-split-shift' create two shifted version of the 'split-time' mode.

    An extra channel can be added with the flag 'mask', this mask represent which pixels
    are data from the camera.

    Parameters
    ----------
    charge : `numpy.ndarray`
        'charge' pixel values from the observation.
    peakpos : `numpy.ndarray`
        'peakpos' pixel values from the observation.
    telescope_type : `str`
        Telescope type.
    mode : `str`, optional
        Image transformation mode: ('simple', 'simple-shift', 'time-split', 'time-split-shift').
        This define the result_image. (default='simple')
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    Returns
    -------
    numpy.array or tuple of numpy.array
        result_image
    """
    if mode == "simple":
        result_image = _simple(charge, peakpos, telescope_type, mask)
    elif mode == "simple-shift":
        result_image = _simple_shift(charge, peakpos, telescope_type, mask)
    elif mode == "time-split":
        result_image = _simple(charge, peakpos, telescope_type, mask)
    elif mode == "time-split-shift":
        result_image = _simple(charge, peakpos, telescope_type, mask)
    else:
        raise ValueError(f"invalid 'mode': {mode}")
    return result_image

def cameras_to_images(cameras, telescopes_type, mode="simple", mask=True):
    """
    Transform the charge and peakpos values into rectangle images.

    The transformation can be done in different modes:
    - 'simple' just return an image two channels (charge and peakpos), 
    where each pixel is vertical aligned. 
    - 'simple-shift' create two version of each image, where odd rows are 
    shifted to right and left, respectively.
    - 'split-time' mode generates just one multichannel image, where the first channel
    is the charge, and the other channels have binary values and represent each value 
    in peakpos, like a one-hot representation. 
    - 'split-time-shift' create two shifted version of the 'split-time' mode.

    An extra channel can be added with the flag 'mask', this mask represent which pixels
    are data from the camera.

    Parameters
    ----------
    cameras : `list` of `tuple` of `numpy.ndarray`
        Containt 'charge' and 'peakpos' pixel values for each observation.
    telescope_type : `list` of `str`
        List of telescope type for each observation.
    mode : `str`, optional
        Image transformation mode: ('simple', 'simple-shift', 'split-time', 'split-time-shift').
        This define the result_image. (default='simple')
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    Returns
    -------
    `list` of `numpy.ndarray` or `list` of `tuple` of `numpy.ndarray`
        result_images
    """
    result_images = []
    for (charge, peakpos), telescope_type in zip(cameras, telescopes_type):
        result_images.append(camera_to_image(charge, peakpos, telescope_type, mode, mask))
    return result_images


"""
Target preprocessing Functions
=============================

Prepare output targets, normalize and standarize values. 
"""

def get_normal_distribution(shape_array, sigmas):
    """Return a matrix with shape 'shape_array' with the multivariate normal distribution with mean in the center."""
    shape = 2*shape_array + 1
    if len(shape_array) == 1:
        pos = np.linspace(-1, 1, shape[0])
        return multivariate_normal.pdf(pos, [0], sigmas)
    elif len(shape_array) == 2:
        x, y = np.mgrid[-1:1:(2/shape[0]), -1:1:(2/shape[1])]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        return multivariate_normal.pdf(pos, [0, 0], [[sigmas[0], 0], [0, sigmas[1]]])
    elif len(shape_array) == 3:
        x, y, z = np.mgrid[-1:1:(2/shape[0]), -1:1:(2/shape[1]), -1:1:(2/shape[2])]
        pos = np.empty(x.shape + (3,))
        pos[:, :, :, 0] = x; pos[:, :, :, 1] = y; pos[:, :, :, 2] = z
        return multivariate_normal.pdf(pos, [0, 0, 0], [[sigmas[0], 0, 0], [0, sigmas[1], 0 ], [0, 0, sigmas[2]]])
    else:
        raise ValueError(f"Only support 1 to 3 dimensions: {len(shape_array)}")

def get_distance(shape_array):
    """Return a matrix with shape 'shape_array' with the distances from the center."""
    shape = 2*shape_array + 1
    if len(shape_array) == 1:
        pos = np.arange(-(shape[0]//2),1+shape[0]//2)
        return pos
    elif len(shape_array) == 2:
        x, y =   np.mgrid[-(shape[0]//2):1+shape[0]//2, -(shape[1]//2):1+shape[1]//2]
        x -= shape[0]//2
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        return np.linalg.norm(pos, axis=-1)
    elif len(shape_array) == 3:
        x, y, z = np.mgrid[-(shape[0]//2):1+shape[0]//2, -(shape[1]//2):1+shape[1]//2, -(shape[2]//2):1+shape[2]//2]
        pos = np.empty(x.shape + (3,))
        pos[:, :, :, 0] = x; pos[:, :, :, 1] = y; pos[:, :, :, 2] = z
        return np.linalg.norm(pos, axis=-1)
    else:
        raise ValueError(f"Only support 1 to 3 dimensions: {len(shape_array)}")

def probability_map(target_values, target_names, target_shapes, target_domains, target_resolutions, target_sigmas):
    """
    Generate a probability cube, where each dimension correspond to each target.

    Each cell represent the probability of each combination of the targets. The 
    size and the value that represent each cell is given by the target_domains and
    target_resolutions.

    config = {
        target_resolutions = {
            "name" : n° of cells,
        },
        target_domains = {
            "name", (min, max)
        }
        target_shapes = [a, b, c] //related with target_names
    }

    Parameters
    ==========
    """
    y = np.zeros((len(target_values), *target_shapes), dtype=float)
    target_domains_arr = np.array(target_domains)
    target_resolutions_arr = np.array(target_resolutions)
    indices = np.floor((target_values - target_domains_arr[:,0])/target_resolutions_arr).astype(int)
    target_shapes_arr = np.array(target_shapes)

    # pre calculated probability map
    if not hasattr(probability_map, "normal_dist"):
        probability_map.normal_dist = get_normal_distribution(shape_array=target_shapes_arr, sigmas=target_sigmas)

    normal_dist_centers = target_shapes_arr
    slices_start = normal_dist_centers - indices
    slices_end = slices_start + target_shapes_arr

    if len(target_names) == 1:
        for i, (slice_start, slice_end) in enumerate(zip(slices_start, slices_end)):
            y[i] = probability_map.normal_dist[slice_start[0]:slice_end[0]]
            y[i] /= y[i].sum()
    elif len(target_names) == 2:
        for i, (slice_start, slice_end) in enumerate(zip(slices_start, slices_end)):
            y[i] = probability_map.normal_dist[slice_start[0]:slice_end[0], slice_start[1]:slice_end[1]]
            y[i] /= y[i].sum()
    elif len(target_names) == 3:
        for i, (slice_start, slice_end) in enumerate(zip(slices_start, slices_end)):
            y[i] = probability_map.normal_dist[slice_start[0]:slice_end[0], slice_start[1]:slice_end[1], slice_start[2]:slice_end[2]]
            y[i] /= y[i].sum()
    return y

def one_cell(target_values, target_names, target_shapes, target_domains, target_resolutions):
    y = np.zeros((len(target_values), *target_shapes), dtype=float)
    target_domains_arr = np.array(target_domains)
    target_resolutions_arr = np.array(target_resolutions)
    indices = np.floor((target_values - target_domains_arr[:,0])/target_resolutions_arr).astype(int)
    #FIX index error
    if len(target_names) == 1:
        y[np.arange(len(y)), indices[:,0]] = 1
    elif len(target_names) == 2:
        y[np.arange(len(y)), indices[:,0], indices[:,1]] = 1
    elif len(target_names) == 3:
        y[np.arange(len(y)), indices[:,0], indices[:,1], indices[:,2]] = 1
    return y

def one_cell_distance(target_values, target_names, target_shapes, target_domains, target_resolutions):
    y = np.zeros((len(target_values), *target_shapes), dtype=float)
    target_domains_arr = np.array(target_domains)
    target_resolutions_arr = np.array(target_resolutions)
    indices = ((target_values - target_domains_arr[:,0])/target_resolutions_arr).astype(int)
    target_shapes_arr = np.array(target_shapes)

    # pre calculated probability map
    if not hasattr(one_cell_distance, "distances"):
        one_cell_distance.distances = get_distance(shape_array=target_shapes_arr)

    centers = target_shapes_arr
    slices_start = centers - indices
    slices_end = slices_start + target_shapes_arr

    if len(target_names) == 1:
        for i, (slice_start, slice_end) in enumerate(zip(slices_start, slices_end)):
            y[i] = one_cell_distance.distances[slice_start[0]:slice_end[0]]
    elif len(target_names) == 2:
        for i, (slice_start, slice_end) in enumerate(zip(slices_start, slices_end)):
            y[i] = one_cell_distance.distances[slice_start[0]:slice_end[0], slice_start[1]:slice_end[1]]
    elif len(target_names) == 3:
        for i, (slice_start, slice_end) in enumerate(zip(slices_start, slices_end)):
            y[i] = one_cell_distance.distances[slice_start[0]:slice_end[0], slice_start[1]:slice_end[1], slice_start[2]:slice_end[2]]
    return y


def two_outputs_probability_map(target_values, target_names, target_to_output, target_shapes, target_domains, target_resolutions):
    raise NotImplementedError

def two_outputs_one_cell(target_values, target_names, target_to_output, target_shapes, target_domains, target_resolutions):
    raise NotImplementedError

def targets_to_matrix(targets_values, 
                      target_names=["alt", "az", "log10_mc_energy"], 
                      target_mode="probability_map",
                      target_mode_config={"target_shapes": (81, 81, 81),
                                          "target_domains": [(1.04, 1.39), (-0.52, 0.52), (-2.351, 2.47)],
                                          "target_resolutions": (0.034999999999, 0.10400000000000001, 0.418),
                                          "target_sigmas": (0.02, 0.02, 0.02)}):
    if target_mode == "lineal":
        return targets_values
    elif target_mode == "probability_map":
        return probability_map(targets_values, target_names, **target_mode_config)
    elif target_mode == "one_cell":
        return one_cell(targets_values, target_names, **target_mode_config)
    elif target_mode == "distance":
        return one_cell_distance(targets_values, target_names, **target_mode_config)
    elif target_mode == "two_outputs_probability_map":
        return two_outputs_probability_map(targets_values, target_names, **target_mode_config)
    elif target_mode == "two_outputs_one_cell":
        return two_outputs_one_cell(targets_values, target_names, **target_mode_config)
    else:
        raise ValueError(f"Invalid target_mode: '{target_mode}'" )