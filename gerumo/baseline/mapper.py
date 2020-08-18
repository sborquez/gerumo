from ctapipe.containers import DL1CameraContainer
from ctapipe.instrument import CameraGeometry, TelescopeDescription, SubarrayDescription, CameraDescription
from pandas import DataFrame

from gerumo import load_camera
import astropy.units as u

cameras = dict()
geometries = dict()
telescope_descriptions = dict()


def get_camera_geometry(cam_name: str) -> CameraGeometry:
    if geometries.get(cam_name) is None:
        geometries[cam_name] = CameraGeometry.from_name(cam_name)
    return geometries[cam_name]


def get_camera(cam_name: str) -> CameraDescription:
    if cameras.get(cam_name) is None:
        cameras[cam_name] = CameraDescription.from_name(cam_name)
    return cameras[cam_name]


def get_telescope_description(optics_name: str, cam_name: str) -> TelescopeDescription:
    if telescope_descriptions.get(optics_name) is None:
        telescope_descriptions[optics_name] = dict()
    if telescope_descriptions[optics_name].get(cam_name) is None:
        telescope_descriptions[optics_name][cam_name] = TelescopeDescription.from_name(optics_name, cam_name)
    return telescope_descriptions[optics_name][cam_name]


def split_tel_type(tel_type: str):
    tel_name = tel_type.split('_')
    optics_name, camera_name = tel_name
    if optics_name[:3] == "SST":
        optics_name = 'SST-' + optics_name[3:]
    return optics_name, camera_name


def generate_subarray_description(dataset: DataFrame, event_id: str) -> SubarrayDescription:
    tel_descriptions = dict()
    tel_positions = dict()
    event_group: DataFrame = dataset.groupby("event_unique_id").get_group(event_id)
    for idx, tel in event_group.iterrows():
        tel_id = tel["telescope_id"]
        tel_name = tel['type']
        optics_name, camera_name = split_tel_type(tel_name)
        tel_descriptions[tel_id] = get_telescope_description(optics_name, camera_name)
        tel_positions[tel_id] = [tel['x'], tel['y'], tel['z']]
    return SubarrayDescription(event_id, tel_positions=tel_positions, tel_descriptions=tel_descriptions)


def generate_dl1_container(dataset: DataFrame, event_id: str, version="ML2") -> DL1CameraContainer:
    tels = dict()
    event_group: DataFrame = dataset.groupby("event_unique_id").get_group(event_id)
    for idx, tel in event_group.iterrows():
        tel_id = tel["telescope_id"]

        dl1 = DL1CameraContainer()
        charge, peakpos = load_camera(tel["source"], tel["folder"], tel["type"], tel_id, version=version)
        dl1.image = charge
        dl1.peak_time = peakpos

        tels[tel_id] = dl1
    return tels