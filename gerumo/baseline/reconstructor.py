from typing import Union, Tuple, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from ctapipe.image import hillas_parameters, timing_parameters, \
    tailcuts_clean, number_of_islands, largest_island
from ctapipe.instrument import CameraGeometry
from ctapipe.reco import HillasReconstructor
from ctapipe.utils import CutFlow
from ctapipe.visualization import CameraDisplay, ArrayDisplay
from pandas import DataFrame
from tqdm import tqdm

from gerumo import load_dataset, load_cameras, TELESCOPES_ALIAS
from gerumo.baseline.cutflow import generate_observation_cutflow, CFO_MIN_PIXEL, CFO_MIN_CHARGE, CFO_POOR_MOMENTS, \
    CFO_CLOSE_EDGE, \
    CFO_BAD_ELLIP, CFE_MIN_TELS_RECO, generate_event_cutflow, CFO_NEGATIVE_CHARGE
from gerumo.baseline.mapper import get_camera_geometry, split_tel_type, generate_subarray_description, get_camera, \
    get_telescope_description
from gerumo.data.io import load_array_direction

cleaning_level = {
    'ASTRICam': (5, 7, 2),
    'LSTCam': (3, 15, 2),
    'FlashCam': (6, 14, 2),
    'DigiCam': (3, 5, 2)
}


def camera_radius(camid_to_efl, cam_id="all"):
    average_camera_radii_deg = {
        "ASTRICam": 4.67,
        "CHEC": 3.93,
        "DigiCam": 4.56,
        "FlashCam": 3.95,
        "NectarCam": 4.05,
        "LSTCam": 2.31
    }

    if cam_id in camid_to_efl.keys():
        foclen_meters = camid_to_efl[cam_id]
        average_camera_radius_meters = (
            np.math.tan(np.math.radians(average_camera_radii_deg[cam_id])) * foclen_meters
        )
    elif cam_id == "all":
        average_camera_radius_meters = 0
    else:
        raise ValueError("Unknown camid", cam_id)
    return average_camera_radius_meters


def mask_from_biggest_island(charge: np.array, geometry: CameraGeometry, mask):
    n_islands, labels = number_of_islands(geometry, mask)

    if n_islands == 1:
        camera_biggest = geometry[mask]
        charge_biggest = charge[mask]
    elif n_islands > 1:
        mask_biggest = largest_island(labels)
        camera_biggest = geometry[mask_biggest]
        charge_biggest = charge[mask_biggest]
    else:
        camera_biggest = geometry
        charge_biggest = charge
    return charge_biggest, camera_biggest


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

    mask = tailcuts_clean(
        geometry,
        charge,
        boundary_thresh=boundary_threshold,
        picture_thresh=picture_threshold
    )
    new_charge = np.copy(charge)
    new_charge[~mask] = 0
    return new_charge, mask


def get_observation_parameters(charge: np.array, peak: np.array, cam_name: str, cutflow: CutFlow,
                               boundary_threshold: float = None,
                               picture_threshold: float = None,
                               min_neighbours: float = None,
                               plot: bool = False
                               ):
    """
    :param charge: Charge image
    :param peak: Peak time image
    :param cam_name: Camera name. e.g. FlashCam, ASTRICam, etc.
    :param cutflow: Cutflow for selection
    :param boundary_threshold: (Optional) Cleaning parameter: boundary threshold
    :param picture_threshold: (Optional) Cleaning parameter: picture threshold
    :param min_neighbours: (Optional) Cleaning parameter: minimum neighbours
    :param plot: If True, for each observation a plot will be shown (Default: False)
    :return: hillas containers, leakage container, number of islands, island IDs, timing container, timing gradient
    """
    charge_biggest, mask = clean_charge(charge, cam_name,
                                        boundary_threshold, picture_threshold, min_neighbours)

    camera = get_camera(cam_name)
    geometry = camera.geometry
    charge_biggest, camera_biggest = mask_from_biggest_island(charge, geometry, mask)
    if cutflow.cut(CFO_MIN_PIXEL, charge_biggest):
        return
    if cutflow.cut(CFO_MIN_CHARGE, np.sum(charge_biggest)):
        return
    if cutflow.cut(CFO_NEGATIVE_CHARGE, charge_biggest):
        return

    leakages = {}
    """
    if np.sum(charge_biggest[mask]) != 0.0:
        leakage_biggest = leakage(geometry, charge_biggest, mask)
        leakages["leak1_reco"] = leakage_biggest["leakage1_intensity"]
        leakages["leak2_reco"] = leakage_biggest["leakage2_intensity"]
    else:
        leakages["leak1_reco"] = 0.0
        leakages["leak2_reco"] = 0.0
    """

    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        CameraDisplay(geometry, charge, ax=ax1).add_colorbar()
        CameraDisplay(camera_biggest, charge_biggest, ax=ax2).add_colorbar()
        plt.show()

    moments = hillas_parameters(camera_biggest, charge_biggest)
    if cutflow.cut(CFO_POOR_MOMENTS, moments):
        return
    if cutflow.cut(CFO_CLOSE_EDGE, moments, camera.camera_name):
        return
    if cutflow.cut(CFO_BAD_ELLIP, moments):
        return

    timing_c = timing_parameters(geometry, charge, peak, moments, mask)
    time_gradient = timing_c.slope.value if geometry.camera_name != 'ASTRICam' else moments.skewness
    return moments, leakages, timing_c, time_gradient


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

        self.cameras_by_event = dict((event_id, []) for event_id in self.event_uids)
        self.n_tels_by_event = {}
        for event_id, tel_type, tel_id, (charge, peak) in zip(
                self.dataset["event_unique_id"],
                self.dataset["type"],
                self.dataset["telescope_id"],
                load_cameras(self.dataset, version=version)
        ):
            if event_id not in self.n_tels_by_event:
                self.n_tels_by_event[event_id] = {}
            if tel_type not in self.n_tels_by_event[event_id]:
                self.n_tels_by_event[event_id][tel_type] = 1
            else:
                self.n_tels_by_event[event_id][tel_type] += 1
            self.cameras_by_event[event_id].append((tel_id, tel_type, charge, peak))

    @property
    def event_uids(self) -> list:
        return self.dataset["event_unique_id"].unique()

    @property
    def hdf5_files(self) -> np.ndarray:
        return (self.dataset["folder"] + "/" + self.dataset["source"]).unique()

    def get_event_hdf5_file(self, event_id: str, tel_id: str):
        event_group = self.dataset.groupby("event_unique_id").get_group(event_id)
        tel_row: DataFrame = event_group.loc[event_group["telescope_id"] == tel_id]
        tel_row = tel_row.loc[tel_row.index[0]]
        return tel_row["folder"] + "/" + tel_row["source"]

    @property
    def camera_radius(self):
        tel_types: List[str] = self.dataset["type"].unique()
        cam_and_foclens = {}
        for t in tel_types:
            optics_name, camera_name = split_tel_type(t)
            desc = get_telescope_description(optics_name, camera_name)
            cam_and_foclens[camera_name] = desc.optics.equivalent_focal_length.value
        return {cam_name: camera_radius(cam_and_foclens, cam_name) for cam_name in cam_and_foclens.keys()}

    @property
    def mc_values(self):
        values = dict()
        events = self.dataset[
            ["event_unique_id", "alt", "az", "mc_energy", "core_x", "core_y"]].drop_duplicates().set_index(
            ["event_unique_id"])
        for event_uid in self.event_uids:
            event = events.loc[event_uid]
            values[event_uid] = dict(
                alt=event["alt"],
                az=event["az"],
                mc_energy=event["mc_energy"],
                core_x=event["core_x"],
                core_y=event["core_y"]
            )
        return values

    def reconstruct_event(self, event_id: str, event_cutflow: CutFlow, obs_cutflow: CutFlow,
                          plot: bool = False) -> Union[None, dict]:
        hillas_containers = dict()
        time_gradients = dict()

        run_array_direction = None
        n_valid_tels = 0
        for tel_id, tel_type, charge, peak in self.cameras_by_event[event_id]:
            optics_name, camera_name = split_tel_type(tel_type)
            params = get_observation_parameters(charge, peak, camera_name, obs_cutflow, plot=plot)
            if params is None:
                continue
            moments, _, _, time_gradient = params
            hillas_containers[tel_id] = moments
            time_gradients[tel_id] = time_gradient

            hdf5_file = self.get_event_hdf5_file(event_id, tel_id)
            telescope_alias = TELESCOPES_ALIAS[self.version][tel_type]
            run_array_direction = self.array_directions[hdf5_file][telescope_alias][tel_id]
            n_valid_tels += 1
        subarray = generate_subarray_description(self.dataset, event_id)

        if run_array_direction is None:
            return
        if event_cutflow.cut(CFE_MIN_TELS_RECO, len(hillas_containers)):
            return

        array_pointing = SkyCoord(
            az=run_array_direction[0] * u.rad,
            alt=run_array_direction[1] * u.rad,
            frame="altaz"
        )
        reco = self.reconstructor.predict(hillas_containers, subarray, array_pointing)
        mc = self.mc_values
        return dict(
            pred_az=2*np.pi*u.rad + reco.az,
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

    def get_event_cams_and_foclens(self, event_id: str):
        subarray = generate_subarray_description(self.dataset, event_id)
        tel_types = subarray.telescope_types
        return {
            tel_types[i].camera.camera_name: tel_types[i].optics.equivalent_focal_length.value
            for i in range(len(tel_types))
        }

    def reconstruct_all(self, max_events=None,
                        min_valid_observations=2,
                        npix_bounds: Tuple[float, float] = None,
                        charge_bounds: Tuple[float, float] = None,
                        ellipticity_bounds: Tuple[float, float] = None,
                        nominal_distance_bounds: Tuple[float, float] = None,
                        plot: bool = False
                        ) -> dict:
        event_ids = self.event_uids
        if max_events is not None and max_events < len(event_ids):
            event_ids = event_ids[:max_events]

        event_cutflow = generate_event_cutflow(min_valid_observations)
        obs_cutflow = generate_observation_cutflow(self.camera_radius, npix_bounds, charge_bounds,
                                                   ellipticity_bounds, nominal_distance_bounds)

        reconstructions = {}
        for event_id in tqdm(event_ids):
            reco = self.reconstruct_event(event_id, event_cutflow, obs_cutflow, plot=plot)
            if reco is None:
                continue
            reconstructions[event_id] = reco
        print(f"N. Events Reconstructed: {len(reconstructions)}")
        return reconstructions

    def plot_metrics(self, max_events: int = 100, min_valid_observations=2, plot_charges: bool = False,
                     save_to: str = None):
        import ctaplot

        reco = self.reconstruct_all(max_events, min_valid_observations=min_valid_observations, plot=plot_charges)
        preds = list(reco.values())

        reco_alt = np.array([pred['pred_alt'] / (1 * u.rad) for pred in preds])
        reco_az = np.array([pred['pred_az'] / (1 * u.rad) for pred in preds])

        alt = np.array([pred['alt'] / (1 * u.rad) for pred in preds])
        az = np.array([pred['az'] / (1 * u.rad) for pred in preds])
        energy = np.array([pred['mc_energy'] for pred in preds])

        plt.subplots()
        ctaplot.plot_angular_resolution_per_energy(reco_alt, reco_az, alt, az, energy)
        plt.savefig("hillas_reconstruction.png")
        if save_to is not None:
            self.save_predictions(reco, save_to)

    @staticmethod
    def save_predictions(reco: dict, path: str):
        df = DataFrame.from_dict(reco, orient="index")
        df.drop("event_id", axis=1, inplace=True)
        df.to_csv(path, sep=',', index_label="event_id")
