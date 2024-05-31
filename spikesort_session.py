"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
from spikeinterface import extractors as se
from spikeinterface import sorters as ss
from spikeinterface import preprocessing as sp
from spikeinterface import widgets as sw

# %% Global variables
EPHYS_PATH = Path("../data/raw_data/ephys")
KILOSORT_PATH = Path("../data/preprocessed_data/Kilosort")
LFP_PATH = Path("../data/preprocessed_data/LFP")


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


# %% Top level function
def run_ephys_preprocessing():
    ephys_paths_df = get_ephys_paths_df()
    valid_ephys_paths_df = (
        ephys_paths_df.query(  # cannot run spikesorting on multiblock recordings or recordings missing data
            "multiblock_recording == False and data_missing == False and spikeinterface_readable == True"
        )
    )
    for vp in valid_ephys_paths_df.iterrows():
        pass
    return


# %%
def spikesort_session():
    return


# %% Extract LFP


def preprocess_session_LFP(
    ephys_path,
    LFP_stream_name,
    channel_downsample_factor=8,
    bandpass_max=450,
    downsample_frequency=1000,
):
    """
    Function to extract and process LFP data from raw open-ephys data.
    Steps: 1) Load raw LFP data with spikeinterface
           2) Subselect channels from neuropixel probe. Defualt = 8 which is every 80um vertically
           3) Bandpass filter LFP data (also de-means data)
           4) Downsample LFP data
           5) Convert LFP data to float16, units = uV
    Args:
        ephys_path (Path): Path to raw ephys data folder
        LFP_stream_name (str): Name of LFP stream in open-ephys data
        channel_downsample_factor (int): Factor to downsample channels by
        bandpass_max (int): Maximum frequency for bandpass filter
        downsample_frequency (int): Frequency to downsample LFP data to
    """
    # load and preprocess LFP data with spikeinterface
    raw_LFP = se.read_openephys(ephys_path, stream_name=LFP_stream_name)
    downchanneled_LFP = raw_LFP.channel_slice(channel_ids=raw_LFP.get_channel_ids()[::channel_downsample_factor])
    bp_recording_LFP = sp.bandpass_filter(recording=downchanneled_LFP, freq_min=0.1, freq_max=bandpass_max)
    downsampled_LFP = sp.resample(recording=bp_recording_LFP, resample_rate=downsample_frequency)
    lfp_np32 = downsampled_LFP.get_traces(return_scaled=True)  # units = uV
    lfp_np16 = lfp_np32.astype(np.float16)  # minimal loss of precision while decreasing file size
    return lfp_np16


# %% Functions


def get_ephys_paths_df():
    """
    Returns a pandas DataFrame with data extracted from raw ephys data folders.

    The function iterates over all raw ephys data folders and makes the following checks:
        - Has spike sorting been completed?
            Checks the data/preprocessed_data/Kilosort to see if the session with the same subject and datetime exists.
            See spike_sorting_completed column, boolian.
        - Has LFP been extracted?
            Checks the data/preprocessed_data/LFP to see if the session with the same subject and datetime exists.
            See LFP_extracted column, boolian.
        - Is the recording split into multiple parts?
            If there were any errors during recording, openephys maybe have been restarted and the recording split into different experiments_numbers
            or recording_numbers. To processes these recordings separately to the standard pipeline.
        - Is there missing data?
            In some recording error instances, open_ephys output folders and incomplete and the crutial .dat files are missing. This is flagged in the
            data_missing columns (minority of sessions).
        - Is the data readable with spikeinterface?
            Finds the correct AP and LFP streams in the open-ephys data and tries to read the data with spikeinterface. If the data is readable, the
            is good to go for spike sorting and LFP extraction. If not, the data is flagged as unreadable and the session needs to be processed
            separately to the standard pipeline. This can occur if recordings were quickly aborted and timestamp.npy files are empty. See column
            spikeinterface_readable, boolian.
    """
    all_ephys_paths = [f for d in EPHYS_PATH.iterdir() if d.is_dir() for f in d.iterdir() if f.is_dir()]
    ephys_paths_info = []
    for ephys_path in all_ephys_paths:
        print(ephys_path)
        subject, datetime_string = ephys_path.parts[-2:]
        datetime = dt.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
        spike_sorting_completed = (KILOSORT_PATH / subject / datetime_string).is_dir()
        lfp_extracted = (LFP_PATH / subject / datetime_string).is_dir()
        # define info dict here:
        path_info = {
            "subject": subject,
            "datetime": datetime,
            "ephys_path": ephys_path,
            "spike_sorting_completed": spike_sorting_completed,
            "LFP_extracted": lfp_extracted,
            "AP_stream_name": None,
            "LFP_stream_name": None,
            "multiblock_recording": None,
            "spikeinterface_readable": None,
            "data_missing": None,
        }
        # work through open-ephys folder structure to determine sorting errors
        record_nodes = [f for f in ephys_path.iterdir() if f.is_dir()]
        if len(record_nodes) != 1:
            raise ValueError(f"More than one record node found in ephys folder: {ephys_path}")
        record_node_path = record_nodes[0]
        experiment_paths = [f for f in record_node_path.iterdir() if f.is_dir()]
        if len(experiment_paths) != 1:
            print(f"Multiblock Recording: {record_node_path}")
            path_info["multiblock_recording"] = True
            ephys_paths_info.append(path_info)
            continue
        else:
            path_info["multiblock_recording"] = False
            experiment_path = experiment_paths[0]
        recording_paths = [f for f in experiment_path.iterdir() if f.is_dir()]
        if len(recording_paths) != 1:
            print(f"Multiblock Recording: {record_node_path}")
            path_info["multiblock_recording"] = True
            ephys_paths_info.append(path_info)
            continue
        else:
            path_info["multiblock_recording"] = False
            recording_path = recording_paths[0]
        continous_data_path = recording_path / "continuous"
        recording_settings_path = recording_path / "structure.oebin"
        if not continous_data_path.is_dir() or not recording_settings_path.is_file():
            print(f"Missing critical data in folder: {recording_path}")
            path_info["data_missing"] = True
            ephys_paths_info.append(path_info)
            continue
        else:
            path_info["data_missing"] = False
        dat_folders = [f for f in continous_data_path.iterdir()]  # folders containing subject .dat files
        # When multiple subjects are recorded in the same session, the dat folders will contain multiple subjects
        # but one subject will have empty files.
        # Check each subject has an AP and LFP stream, if one is missing - indicative of recording error
        recorded_subjects = np.unique([f.name.split(".")[1].split("-")[0] for f in dat_folders])
        expected_dat_folders = [
            folder for s in recorded_subjects for folder in (f"Neuropix-PXI-100.{s}-AP", f"Neuropix-PXI-100.{s}-LFP")
        ]
        if not all([f in [i.name for i in dat_folders] for f in expected_dat_folders]):
            print(f"Missing critical data in folder: {recording_path}")
            path_info["data_missing"] = True
            ephys_paths_info.append(path_info)
            continue
        else:
            path_info["data_missing"] = False
        AP_stream_name = None
        LFP_stream_name = None
        for f in dat_folders:
            try:
                subject_id, data_type = f.name.split(".")[1].split("-")
            except IndexError:
                raise ValueError(f"Unknown file structure in dat folder: {f}")
            if subject_id == subject.replace("_", ""):  # no understores in openephys probe names
                stream_id = f"{record_node_path.name}#Neuropix-PXI-100.{subject_id}-{data_type}"
                if data_type == "AP":
                    AP_stream_name = stream_id
                elif data_type == "LFP":
                    LFP_stream_name = stream_id
                else:
                    raise ValueError(f"Unknown data type in dat folder: {f}")
        if not AP_stream_name:
            raise ValueError(f"subject AP stream not found in ephys folder: {recording_path}")
        if not LFP_stream_name:
            raise ValueError(f"subject LFP stream not found in ephys folder: {recording_path}")
        try:  # validate that open-ephys data can be read with spike interface
            se.read_openephys(ephys_path, stream_name=AP_stream_name)
            se.read_openephys(ephys_path, stream_name=LFP_stream_name)
            path_info["spikeinterface_readable"] = True
            path_info["AP_stream_name"] = AP_stream_name
            path_info["LFP_stream_name"] = LFP_stream_name
        except ValueError:
            print(f"Error reading open-ephys data in folder: {ephys_path}")
            path_info["spikeinterface_readable"] = False
        except IndexError:  # timestamps file empty (eg, aborted recording)
            print(f"Error reading open-ephys data in folder: {ephys_path}")
            path_info["spikeinterface_readable"] = False
        ephys_paths_info.append(path_info)
    return pd.DataFrame(ephys_paths_info)
