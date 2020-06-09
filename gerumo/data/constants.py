"""
Data and dataset constants
==========================
"""
from os import path
from astropy.units.cds import rad, deg, eV
import numpy as np


__all__ = [ 
    'TELESCOPES', 'TELESCOPE_FEATURES', 
    'TARGETS', 'TARGET_UNITS',
    'IMAGES_SIZE', 'INPUT_SHAPE', 'PIXELS_POSITION'
]


# Telescopes types
TELESCOPES = ["LST_LSTCam", "MST_FlashCam", "SST1M_DigiCam"]

# Telescope array informatin
TELESCOPE_FEATURES = ["x", "y"]

# Regression Targets
TARGETS = ["alt", "az", "mc_energy", "log10_mc_energy"]
TARGET_UNITS = [rad, rad, 1e12*eV, np.log10(1e12)*eV]

# IMAGES Size
IMAGES_SIZE = {
    "LST_LSTCam" : (55, 47),
    "MST_FlashCam": (84, 29),
    "SST1M_DigiCam": (72, 25)
}

INPUT_SHAPE = {
    "simple": {
        "LST_LSTCam" : (55, 47, 2),
        "MST_FlashCam": (84, 29, 2),
        "SST1M_DigiCam": (72, 25, 2) 
    },
    "simple-shift": {
        "LST_LSTCam" : (2, 55, 47, 2),
        "MST_FlashCam": (2, 84, 29, 2),
        "SST1M_DigiCam": (2, 72, 25, 2) 
    },
    "simple-mask": {
        "LST_LSTCam" : (55, 47, 3),
        "MST_FlashCam": (84, 29, 3),
        "SST1M_DigiCam": (72, 25, 3)
    },
    "simple-shift-mask": {
        "LST_LSTCam" : (2, 55, 47, 3),
        "MST_FlashCam": (2, 84, 29, 3),
        "SST1M_DigiCam": (2, 72, 25, 3) 
    },
}

pixpos_folder = path.join(path.dirname(__file__), "pixels_positions")
PIXELS_POSITION = {}
for mode in ("simple", "simple_shift"): #time_split, time_split_shift):
    PIXELS_POSITION[mode] = {}
    for telescope in TELESCOPES:
        PIXELS_POSITION[mode][telescope] = np.loadtxt(path.join(pixpos_folder, mode, f"{telescope}.npy"), dtype=float).astype(int)
