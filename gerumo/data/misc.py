"""
Utilities functions
===================
"""

from math import ceil
from os.path import join
import numpy as np
import tables
from .dataset import _telescope_table, _telescopes_info_attributes
from . import TELESCOPES, TELESCOPES_ALIAS

__all__ = [
    'get_resolution', 'get_shape', 
    'extract_pixel_positions',
]


"""
Misc Functions
===============
"""

def get_resolution(targets, targets_domain, targets_shape):
    """Return the targets resolution for each target given the targets shape"""
    targets_resolution = {}
    for target in targets:
        vmin, vmax = targets_domain[target]
        shape = targets_shape[target]
        targets_resolution[target]  = (vmax -vmin) / shape
    return targets_resolution


def get_shape(targets, targets_domain, targets_resolution):
    """Return the targets shape for each target given the targets resolution"""
    targets_shape = {}
    for target in targets:
        vmin, vmax = targets_domain[target]
        resolution = targets_resolution[target]
        targets_shape[target]  = ceil((vmax -vmin) / resolution)
    return targets_resolution


"""
Pixel Positions Functions
===============
"""

def LST_LSTCam_align(pixels_position_array):
    xs = pixels_position_array[0,:]
    ys = pixels_position_array[1,:]
    # Distance matrix:
    delta_x = np.array([xs])-np.array([xs]).T
    delta_y = np.array([ys])-np.array([ys]).T
    dists = (delta_x**2+delta_y**2)**0.5
    angles = np.arctan2(delta_y, delta_x) # Angles from -pi to pi
    # Binary search, find maximum radious where no cell has more than 6 neighbors
    rad1 = 0
    rad2 = np.max(dists)
    for i in range(1000):
        rad = (rad1+rad2)/2.0
        neighs = dists<rad # matrix with true if i,j are neighbors
        np.fill_diagonal(neighs,False)
        max_neighs = np.max(np.sum(neighs,axis=1))
        if max_neighs>6:
            rad2 = rad
        else:
            rad1 = rad
    #
    rad = rad1
    neighs = dists<rad
    # Get a group of angles on an interval:
    ang_start = 0
    ang_end = np.pi*(6//2)
    # Neighbors with angle between those two
    conditions = np.all([neighs,angles>=ang_start,angles<ang_end],axis=0)
    neighbors = np.where(conditions)
    neigh_angles = angles[neighbors]
    # From the angles in this group, pick the median as the main axis
    main_axis_ang = np.median(neigh_angles)
    main_x = np.cos(main_axis_ang)
    main_y = np.sin(main_axis_ang)
    # Apply transformation
    tx = xs*main_x+ys*main_y
    ty = xs*main_y-ys*main_x
    # Now compute the maximum separation between neighboors in the main axis.
    dx = np.max(delta_x[neighs]*main_x+delta_y[neighs]*main_y)
    # Scale main axis by half of that separation:
    tx = np.round(tx/(dx/2.0))
    # Now compute the maximum separation between neighboors in the secondary axis.
    dy = np.max(delta_x[neighs]*main_y-delta_y[neighs]*main_x)
    # Scale secondary axis by that separation:
    ty = np.round(ty/dy)
    return np.stack((tx, ty))


def to_simple_and_shift(pixels_position_array):
    # get pixels positions
    xs = pixels_position_array[0]
    ys = pixels_position_array[1]
    # indices of x and y pixels position
    i = np.arange(0, len(ys))
    # row values of the telescope
    y_levels = np.sort(np.unique(ys))
    # image dimension
    nrows = len(y_levels)
    ncols = len(np.unique(xs))//2 + 1
    # new translated pixel positions
    new_x_l = np.copy(xs) # new pixels x positions left shift
    new_x_r = np.copy(xs) # new pixels x positions right shift
    new_y = np.copy(ys)
    # shift odd rows
    dx = 0
    for level, y_value in enumerate(y_levels):
        indices = i[ys == y_value]
        if dx == 0:
            dx = np.diff(np.sort(xs[indices])).min()/2
        if level % 2 != 0:
            new_x_l[indices] -= dx
            new_x_r[indices] += dx
    # round values
    new_x_l = np.round(new_x_l, 3)
    new_x_r = np.round(new_x_r, 3)
    # max indices of image output
    max_col_l = len(np.unique(new_x_l)) - 1
    max_col_r = len(np.unique(new_x_r)) - 1
    max_row = nrows - 1
    # apply linear transfomation
    new_x_l = ((max_col_l/(new_x_l.max() - new_x_l.min())) * (new_x_l - new_x_l.min()))
    new_x_l = np.round(new_x_l).astype(int)
    new_x_r = ((max_col_r/(new_x_r.max() - new_x_r.min())) * (new_x_r - new_x_r.min()))
    new_x_r = np.round(new_x_r).astype(int)
    new_y = ((max_row/(new_y.max() - new_y.min())) * (new_y - new_y.min()))
    new_y = np.round(new_y).astype(int)
    # prepare output
    simple = np.vstack((new_x_r, new_y))
    simple_shift = np.vstack((new_x_l, new_x_r, new_y))
    return simple, simple_shift


def extract_pixel_positions(hdf5_filepath, pixpos_folder, version="ML2"):
    """Extract and save from file the information about pixel position.
    Extract and apply transformation  to pixel position of each telescope 
    type and for each camera_to_image 'mode'. Saves it to use them in the
    'camera_to_image' function.

    Note: This function is used just once. The pixel position can be shared
    in a numpy file format.
    """
    inverse_alias = {TELESCOPES_ALIAS[version][t]:t for t in TELESCOPES}
    modes = ('raw', 'simple', 'simple-shift')
    pixpos_folder = join(pixpos_folder, version)
    print(pixpos_folder)

    hdf5_file = tables.open_file(hdf5_filepath, "r")
    telescopes_info = hdf5_file.root[_telescope_table[version]]

    all_pixpos = {}
    
    # Extract pixel position array from hdf5 file
    raw_pixpos = {}
    for telescope in telescopes_info:
        type_ = telescope[_telescopes_info_attributes[version]["type"]].decode("utf-8")
        if type_ not in inverse_alias:
            continue
        num_pixels = telescope[_telescopes_info_attributes[version]["num_pixels"]]
        type_ = inverse_alias[type_]
        if version == "ML2":
            raw_pixpos[type_] = telescope[_telescopes_info_attributes[version]["pixel_pos"]][:num_pixels, :].T
        else:
            raw_pixpos[type_] = telescope[_telescopes_info_attributes[version]["pixel_pos"]][:, :num_pixels]
    
    # Save raw pixpos
    all_pixpos['raw'] = {}
    for telescope, pixpos in raw_pixpos.items():
        if telescope == "LST_LSTCam":
            LST_LSTCam_not_aligm = pixpos
            all_pixpos['raw']['LST_LSTCam_not_aligm'] = LST_LSTCam_not_aligm
            raw_pixpos["LST_LSTCam"] = LST_LSTCam_align(pixpos)
            pixpos = raw_pixpos["LST_LSTCam"]
            np.savetxt(join(pixpos_folder,'raw', f'{telescope}_not_align.npy'), LST_LSTCam_not_aligm)
        np.savetxt(join(pixpos_folder,'raw', f'{telescope}.npy'), pixpos)
        all_pixpos['raw'][telescope] = pixpos

    # Generate simple align and shift align
    all_pixpos['simple'] = {}
    all_pixpos['simple_shift'] = {}
    for telescope, pixpos in raw_pixpos.items():
        simple, shift = to_simple_and_shift(pixpos)
        all_pixpos['simple'][telescope] = simple
        all_pixpos['shift'][telescope] = shift
        np.savetxt(join(pixpos_folder,'simple', f'{telescope}.npy'), simple)
        np.savetxt(join(pixpos_folder,'simple_shift', f'{telescope}.npy'), shift)
    
    return all_pixpos
