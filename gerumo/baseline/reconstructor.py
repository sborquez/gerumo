from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from ctapipe.image import hillas_parameters, leakage, number_of_islands, timing_parameters, \
    tailcuts_clean
from ctapipe.reco import HillasReconstructor
from pandas import DataFrame
from tqdm import tqdm

from gerumo import load_dataset, load_cameras, TELESCOPES_ALIAS
from gerumo.baseline.mapper import get_camera_geometry, split_tel_type, generate_subarray_description
from gerumo.data.io import load_array_direction

cleaning_level = {
    'ASTRICam': (5, 7, 2),
    'LSTCam': (3.5, 7.5, 2),
    'FlashCam': (4, 8, 2),
    'DigiCam': (4, 8, 2)  # TODO: Define this parameters
}


def clean_charge(charge: np.array, cam_name: str,
                 boundary_threshold: float = None,
                 picture_threshold: float = None,
                 min_neighbours: float = None):
    geometry = get_camera_geometry(cam_name)
    clean_default_params = cleaning_level.get(cam_name)
    if boundary_threshold is None:
        boundary_threshold = clean_default_params[0]
    if picture_threshold is None:
        picture_threshold = clean_default_params[1]
    if min_neighbours is None:
        min_neighbours = clean_default_params[2]

    return tailcuts_clean(
        geometry,
        charge,
        boundary_thresh=boundary_threshold,
        picture_thresh=picture_threshold,
        min_number_picture_neighbors=min_neighbours
    )


def get_observation_parameters(charge: np.array, peak: np.array, cam_name: str,
                               max_leakage: float = 0.2,
                               boundary_threshold: float = None,
                               picture_threshold: float = None,
                               min_neighbours: float = None):
    """
    :param charge: Charge image
    :param peak: Peak time image
    :param cam_name: Camera name. e.g. FlashCam, ASTRICam, etc.
    :param max_leakage: Maximum allowed leakage. If the leakage is higher, no timing parameters will be returned
    :param min_neighbours: (Optional) Cleaning parameter: minimum neighbours
    :param picture_threshold: (Optional) Cleaning parameter: picture threshold
    :param boundary_threshold: (Optional) Cleaning parameter: boundary threshold
    :return: hillas containers, leakage container, number of islands, island IDs, timing container, timing gradient
    """
    clean = clean_charge(charge, cam_name, boundary_threshold, picture_threshold, min_neighbours)
    if clean.sum() < 5:
        return None, None, None, None, None, None

    geometry = get_camera_geometry(cam_name)
    hillas_c = hillas_parameters(geometry[clean], charge[clean])
    leakage_c = leakage(geometry, charge, clean)
    n_islands, island_ids = number_of_islands(geometry, clean)

    if leakage_c.intensity_width_2 > max_leakage:
        return hillas_c, leakage_c, n_islands, island_ids, None, None

    timing_c = timing_parameters(geometry, charge, peak, hillas_c, clean)
    time_gradient = timing_c.slope.value if geometry.camera_name != 'ASTRICam' else hillas_c.skewness
    return hillas_c, leakage_c, n_islands, island_ids, timing_c, time_gradient
    # if abs(time_gradient) < 0.2:
    #     time_gradient = 1.0


class Reconstructor:
    def __init__(self, events_path: str, telescopes_path: str, version="ML1"):
        if version == "ML2":
            raise NotImplementedError("This reconstructor is not implemented to work with ML2 yet")
        self.version = version
        self.dataset = load_dataset(events_path, telescopes_path)
        self.reconstructor = HillasReconstructor()
        self.array_directions = dict()
        for hdf5_file in self.hdf5_files:
            self.array_directions[hdf5_file] = load_array_direction(hdf5_file)

        self.cameras_by_event = dict((event_id, []) for event_id in self.event_ids)
        for event_id, tel_type, tel_id, (charge, peak) in zip(
                self.dataset["event_unique_id"],
                self.dataset["type"],
                self.dataset["telescope_id"],
                load_cameras(self.dataset, version=version)
        ):
            self.cameras_by_event[event_id].append((tel_id, tel_type, charge, peak))

    @property
    def event_ids(self) -> list:
        return self.dataset["event_unique_id"].to_list()

    @property
    def hdf5_files(self) -> np.ndarray:
        return (self.dataset["folder"] + "/" + self.dataset["source"]).unique()

    def get_event_hdf5_file(self, event_id: str, tel_id: str):
        event_group = self.dataset.groupby("event_unique_id").get_group(event_id)
        tel_row: DataFrame = event_group.loc[event_group["telescope_id"] == tel_id]
        tel_row = tel_row.loc[tel_row.index[0]]
        return tel_row["folder"] + "/" + tel_row["source"]

    @property
    def mc_values(self):
        values = dict()
        events = self.dataset[
            ["event_unique_id", "alt", "az", "mc_energy", "core_x", "core_y"]].drop_duplicates().set_index(
            ["event_unique_id"])
        for event_id in self.event_ids:
            event = events.loc[event_id]
            values[event_id] = dict(
                alt=event["alt"],
                az=event["az"],
                mc_energy=event["mc_energy"],
                core_x=event["core_x"],
                core_y=event["core_y"]
            )
        return values

    def reconstruct_event(self, event_id: str, min_valid_observations=2) -> Union[None, dict]:
        hillas_containers = dict()
        time_gradients = dict()

        run_array_direction = None
        for tel_id, tel_type, charge, peak in self.cameras_by_event[event_id]:
            optics_name, camera_name = split_tel_type(tel_type)
            hillas_c, _, _, _, _, time_gradient = get_observation_parameters(charge, peak, camera_name)
            if hillas_c is None or time_gradient is None:
                continue
            hillas_containers[tel_id] = hillas_c
            time_gradients[tel_id] = time_gradient

            hdf5_file = self.get_event_hdf5_file(event_id, tel_id)
            telescope_alias = TELESCOPES_ALIAS[self.version][tel_type]
            run_array_direction = self.array_directions[hdf5_file][telescope_alias][tel_id]
        subarray = generate_subarray_description(self.dataset, event_id)

        if len(hillas_containers) < min_valid_observations or run_array_direction is None:
            return

        horizon_frame = AltAz()
        array_pointing = SkyCoord(
            az=run_array_direction[0] * u.rad,
            alt=run_array_direction[1] * u.rad,
            frame=horizon_frame
        )
        reco = self.reconstructor.predict(hillas_containers, subarray, array_pointing)
        mc = self.mc_values
        return dict(
            pred_az=reco.az,
            pred_alt=reco.alt,
            pred_core_x=reco.core_x,
            pred_core_y=reco.core_y,
            alt=mc[event_id]["alt"] * u.rad,
            az=mc[event_id]["az"] * u.rad,
            core_x=mc[event_id]["core_x"],
            core_y=mc[event_id]["core_y"],
            mc_energy=mc[event_id]["mc_energy"],
            event_id=event_id
        )

    def reconstruct_all(self, max_events=None, min_valid_observations=2) -> dict:
        event_ids = self.event_ids
        if max_events is not None:
            event_ids = event_ids[:max_events]
        reconstructions = list()
        for event_id in tqdm(event_ids[:max_events]):
            reconstructions.append((event_id, self.reconstruct_event(event_id, min_valid_observations)))
        return dict((event_id, reco) for event_id, reco in reconstructions if reco is not None)

    def plot_metrics(self, max_events: int = 100, min_valid_observations=2):
        import ctaplot
        import matplotlib.pyplot as plt

        preds = list(self.reconstruct_all(max_events, min_valid_observations).values())
        reco_alt = np.array([pred['pred_alt'] / (1 * u.rad) for pred in preds])
        reco_az = np.array([pred['pred_az'] / (1 * u.rad) for pred in preds])
        alt = np.array([pred['alt'] / (1 * u.rad) for pred in preds])
        az = np.array([pred['alt'] / (1 * u.rad) for pred in preds])
        energy = np.array([pred['mc_energy'] for pred in preds])
        fig, ax = plt.subplots()
        ctaplot.plot_angular_resolution_per_energy(reco_alt, reco_az, alt, az, energy)
        plt.savefig("hillas_reconstruction.png")
