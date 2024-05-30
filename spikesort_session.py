"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
from spikeinterface import extractors as se
from spikeinterface import sorters as ss

# %% Global variables
EPHYS_PATH = Path("../data/raw_data/ephys")
KILOSORT_PATH = Path("../data/preprocessed_data/Kilosort")


# %% Functions


# def spikesort_session(ephys_path, specified_output_folder=False):
#     """
#     Function takes an ephys_path as input and spikesorts the session. Currently the function
#     run default Kilosort4 sorting, but can be extended to run non-kilosort preprocessing etc.
#     """
#     # define output folder:
#     subject, collection_datetime = Path(ephys_path).parts[-2:]
#     sortname = specified_output_folder if specified_output_folder else dt.now().isoformat()
#     output_folder = Path(KILOSORT_FOLDER) / subject / collection_datetime / sortname
#     # Load data with spikeinterface
#     raw_AP = se.read_openephys(ephys_path, stream_id="0")  # stream_id 0 is AP data, 1 is LFP data
#     sorting_params = ss.get_default_sorter_params("kilosort4")
#     # Run kilosort4 with defualt parameters
#     sorting = ss.run_sorter(
#         sorter_name="kilosort4", recording=raw_AP, output_folder=str(output_folder), **sorting_params
#     )
#     return


# %% cleaned up version of get ephys paths df


def get_ephys_paths_df():
    """
    Returns a pandas DataFrame with data extracted from raw ephys data folders.
    """
    all_ephys_paths = [f for d in EPHYS_PATH.iterdir() if d.is_dir() for f in d.iterdir()]
    ephys_paths_info = []
    for ephys_path in all_ephys_paths:
        print(ephys_path)
        subject, datetime_string = ephys_path.parts[-2:]
        datetime = dt.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
        spike_sorting_completed = True if (KILOSORT_PATH / subject / datetime_string).is_dir() else False
        # define info dict here:
        path_info = {
            "subject": subject,
            "datetime": datetime,
            "ephys_path": ephys_path,
            "AP_stream_id": None,
            "LFP_stream_id": None,
            "spike_sorting_completed": spike_sorting_completed,
            "recording_restarted": None,
            "readable": None,
            "data_missing": None,
        }
        # work through open-ephys folder structure to determine sorting errors
        record_nodes = [f for f in ephys_path.iterdir() if f.is_dir()]
        assert len(record_nodes) == 1, f"More than one record node found in ephys folder: {ephys_path}"
        record_node = record_nodes[0].name
        experiment_paths = [f for f in record_nodes[0].iterdir() if f.is_dir()]
        recording_paths = [f for f in experiment_paths[0].iterdir() if f.is_dir()]
        if len(experiment_paths) > 1 or len(recording_paths) > 1:  # indicative of recording restart due to buffer error
            path_info["recording_restarted"] = True
            ephys_paths_info.append(path_info)
            continue
        recording_path = recording_paths[0] / "continuous"
        recording_settings_path = recording_paths[0] / "structure.oebin"
        if not recording_path.is_dir() or not recording_settings_path.is_file():
            print(f"Missing data in folder: {recording_path}")
            path_info["data_missing"] = True
            ephys_paths_info.append(path_info)
            continue
        dat_folders = [f for f in recording_path.iterdir()]  # folders containing subject .dat files
        recorded_subjects = np.unique([f.name.split(".")[1].split("-")[0] for f in dat_folders])
        # check each subject has an AP and LFP stream, if one is missing - indicative of recording error
        expected_dat_folders = [
            folder for s in recorded_subjects for folder in (f"Neuropix-PXI-100.{s}-AP", f"Neuropix-PXI-100.{s}-LFP")
        ]
        if not all([f in [i.name for i in dat_folders] for f in expected_dat_folders]):
            print(f"Missing data in folder: {recording_path}")
            path_info["data_missing"] = True
            ephys_paths_info.append(path_info)
            continue
        AP_stream_id = None
        LFP_stream_id = None
        for f in dat_folders:
            try:
                subject_id, data_type = f.name.split(".")[1].split("-")
            except IndexError:
                raise ValueError(f"Unknown file structure in dat folder: {f}")
            if subject_id == subject.replace("_", ""):  # no understores in openephys probe names
                stream_id = f"{record_node}#Neuropix-PXI-100.{subject_id}-{data_type}"
                if data_type == "AP":
                    AP_stream_id = stream_id
                elif data_type == "LFP":
                    LFP_stream_id = stream_id
                else:
                    raise ValueError(f"Unknown data type in dat folder: {f}")
        if not AP_stream_id:
            raise ValueError(f"subject AP stream not found in ephys folder: {recording_path}")
        if not LFP_stream_id:
            raise ValueError(f"subject LFP stream not found in ephys folder: {recording_path}")
        try:  # validate that open-ephys data can be read with spike interface
            se.read_openephys(ephys_path, stream_name=AP_stream_id)
            se.read_openephys(ephys_path, stream_name=LFP_stream_id)
            path_info["readable"] = True
        except ValueError:
            print(f"Error reading open-ephys data in folder: {ephys_path}")
            path_info["readable"] = False
        except IndexError:  # timestamps file empty (aborted recording?)
            print(f"Error reading open-ephys data in folder: {ephys_path}")
            path_info["readable"] = False
        ephys_paths_info.append(path_info)
    return pd.DataFrame(ephys_paths_info)
