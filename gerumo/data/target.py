"""
Target preprocessing Functions
=============================

Prepare output targets, normalize and standarize values. 
"""

from scipy.stats import multivariate_normal
import numpy as np

__all__ = [
    'targets_to_matrix'
]

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