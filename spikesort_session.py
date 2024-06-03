"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
import matplotlib.pyplot as plt
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
def spikesort_session(
    ephys_path,
    AP_stream_name,
    interpolate_bad_channels=True,
    remove_bad_channels=False,
    denoise="destripe",
    save_temp_preprocessed_data=True,
    sorting=False,
    save_qc_figs=True,
):
    """
    Function runs Kilosort4 on a single ephys session, saving the results to the preprocessed_data/Kilosort folder.

    Spike sorting is run with the following preprocessing steps
    - STEP 1 - Phase shift correction (accounts for time between sampling neuropixel data across channels)
    - STEP 2 - Bandpass filter
    - STEP 3 - Detect and remove channels outside the brain
    - STEP 4 - Detect and interpolate bad channels
    - STEP 5 - Denoise with common average referencing or IBL stripe protocol
    - STEP 6 - Motion correction
    - STEP 7 - Save out temp preprocessed data (session Kilosort folder)

    and the following spike sorting steps:
    - STEP 8 - Spike Sorting with Kilosort4 from sacved preprocessed daa
    - STEP 9 - Save out spike sorting results
    - STEP 10 - Remove temp preprocessed data
    """
    # set up output folder
    subject, datetime_string = ephys_path.parts[-2:]
    output_folder = KILOSORT_PATH / subject / datetime_string
    output_folder.mkdir(parents=True, exist_ok=True)
    # preprocess ephys data with spikeinterface
    raw_AP = se.read_openephys(ephys_path, stream_name=AP_stream_name)
    all_channel_ids = raw_AP.get_channel_ids()
    phase_shift_AP = sp.phase_shift(raw_AP)
    highpass_AP = sp.highpass_filter(phase_shift_AP, freq_min=300)
    _, channel_labels = sp.detect_bad_channels(highpass_AP)
    preprocessed_AP = highpass_AP.remove_channels(all_channel_ids[channel_labels == "out"])
    bad_channels = np.concatenate(
        [all_channel_ids[channel_labels == "noise"], all_channel_ids[channel_labels == "dead"]]
    )
    if interpolate_bad_channels:
        preprocessed_AP = sp.interpolate_bad_channels(preprocessed_AP, bad_channel_ids=bad_channels)
    if denoise == "CAR":
        preprocessed_AP = sp.common_reference(preprocessed_AP)
    elif denoise == "destripe":
        preprocessed_AP = sp.highpass_spatial_filter(preprocessed_AP)
    if remove_bad_channels:
        preprocessed_AP = preprocessed_AP.remove_channels(bad_channels)
    if save_qc_figs:
        preprocessing_fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
        sw.plot_traces(raw_AP, backend="matplotlib", clim=(-500, 500), ax=axs[0])
        sw.plot_traces(preprocessed_AP, backend="matplotlib", clim=(-500, 500), ax=axs[1])
        for ax, label in zip(axs, ["Raw", "Preprocessed"]):
            ax.set_tile(label)
            ax.set_xlabel("Time (s)")
        preprocessing_fig.savefig(output_folder / "figs" / "preprocessed_traces.png")
    # Save preprocessed .dat file
    if save_temp_preprocessed_data:
        preprocessed_AP.save(
            folder=output_folder / "preprocessed_temp",
            format="binary",
            n_jobs=40,
            chunk_duration="1s",
            progress_bar=True,
        )
    # Run Kilosort4
    if sorting:
        sorting_params = ss.get_default_sorter_params("kilosort4")
        sorting_params["skip_kilosort_preprocessing"] = True
        sorting_params["do_correction"] = True
        sorting = ss.run_sorter(
            "kilosort4",
            recording=preprocessed_AP,
            output_folder=output_folder / "kilosort4",
            verbose=True,
            **sorting_params,
        )

    return print(f"Session {subject} {datetime_string} spikesorted with Kilosort4")


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
