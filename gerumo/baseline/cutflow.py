import warnings
from collections import OrderedDict
from typing import Tuple

from ctapipe.utils import CutFlow
import numpy as np

CF_NO_CUTS = "noCuts"
CFO_MIN_PIXEL = "min pixel"
CFO_MIN_CHARGE = "min charge"
CFO_POOR_MOMENTS = "poor moments"
CFO_BAD_ELLIP = "bad ellipticity"
CFO_CLOSE_EDGE = "close to the edge"
CFO_NEGATIVE_CHARGE = "negative charge"

CFE_MIN_TELS_TRIG = "min2Tels trig"
CFE_MIN_TELS_RECO = "min2Tels reco"
CFE_DIR_NAN = "direction nan"

CF_DEFAULT = {
    "charge": [50., 1e10],
    "pixel": [3, 1e10],
    "ellipticity": [0.1, 0.6],
    "nominal_distance": [0., 0.8]
}


def _cutflow_set_default(npix_bounds: Tuple[float, float] = None,
                         charge_bounds: Tuple[float, float] = None,
                         ellipticity_bounds: Tuple[float, float] = None,
                         nominal_distance_bounds: Tuple[float, float] = None):
    if npix_bounds is None:
        npix_bounds = list(CF_DEFAULT["pixel"])
    if charge_bounds is None:
        charge_bounds = list(CF_DEFAULT["charge"])
    if ellipticity_bounds is None:
        ellipticity_bounds = list(CF_DEFAULT["ellipticity"])
    if nominal_distance_bounds is None:
        nominal_distance_bounds = list(CF_DEFAULT["nominal_distance"])
    return npix_bounds, charge_bounds, ellipticity_bounds, nominal_distance_bounds


# TODO: Test without filters

def generate_observation_cutflow(camera_radius,
                                 npix_bounds: Tuple[float, float] = None,
                                 charge_bounds: Tuple[float, float] = None,
                                 ellipticity_bounds: Tuple[float, float] = None,
                                 nominal_distance_bounds: Tuple[float, float] = None):
    npix_bounds, charge_bounds, ellipticity_bounds, nominal_distance_bounds = _cutflow_set_default(
        npix_bounds, charge_bounds, ellipticity_bounds, nominal_distance_bounds
    )

    warnings.filterwarnings("ignore", category=FutureWarning)
    cutflow = CutFlow("ImageCutFlow")
    warnings.filterwarnings("default", category=FutureWarning)

    cutflow.set_cuts(
        OrderedDict(
            [
                (CF_NO_CUTS, None),
                (CFO_MIN_PIXEL, lambda s: np.count_nonzero(s) < npix_bounds[0]),
                (CFO_MIN_CHARGE, lambda x: x < charge_bounds[0]),
                (CFO_POOR_MOMENTS, lambda m: m.width <= 0 or m.length <= 0),
                (CFO_BAD_ELLIP,
                 lambda m:
                 (m.width / m.length) < ellipticity_bounds[0] or (m.width / m.length) > ellipticity_bounds[-1],
                 ),
                (CFO_CLOSE_EDGE,
                 lambda m, cam_id:
                 m.r.value > (nominal_distance_bounds[-1] * camera_radius[cam_id]),
                 ),
                (CFO_NEGATIVE_CHARGE,
                 lambda s: np.any(s < 0)
                 )
            ]
        )
    )
    return cutflow


def generate_event_cutflow(min_tels: int = 2):
    warnings.filterwarnings("ignore", category=FutureWarning)
    cutflow = CutFlow("EventCutFlow")
    warnings.filterwarnings("default", category=FutureWarning)

    cutflow.set_cuts(
        OrderedDict(
            [
                (CF_NO_CUTS, None),
                (CFE_MIN_TELS_TRIG, lambda x: x < min_tels),
                (CFE_MIN_TELS_RECO, lambda x: x < min_tels),
                (CFE_DIR_NAN, lambda x: x.is_valid is False),
            ]
        )
    )
    return cutflow
