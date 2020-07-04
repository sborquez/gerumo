"""
Utilities functions
===================
"""

from math import ceil


__all__ = ['get_resolution', 'get_shape']

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

def describe_dataset(dataset):
    files = dataset.source.nunique()
    events = dataset.event_unique_id.nunique()
    obs = len(dataset)
    by_telescope = dataset.type.value_counts()
    print('files', files)
    print('events', events)
    print('observations', obs)
    print('obsevation by telescopes')
    print(by_telescope)