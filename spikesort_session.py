"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
from pathlib import Path
from datetime import datetime as dt
from spikeinterface import extractors as se
from spikeinterface import sorters as ss

# %% Global variables
EPHYS_PATH = Path("../data/raw_data/ephys")
KILOSORT_FOLDER = Path("../data/preprocessed_data/Kilosort")


# %% Functions


def spikesort_session(ephys_path, specified_output_folder=False):
    """
    Function takes an ephys_path as input and spikesorts the session. Currently the function
    run default Kilosort4 sorting, but can be extended to run non-kilosort preprocessing etc.
    """
    # define output folder:
    subject, collection_datetime = Path(ephys_path).parts[-2:]
    sortname = specified_output_folder if specified_output_folder else dt.now().isoformat()
    output_folder = Path(KILOSORT_FOLDER) / subject / collection_datetime / sortname
    # Load data with spikeinterface
    raw_AP = se.read_openephys(ephys_path, stream_id="0")  # stream_id 0 is AP data, 1 is LFP data
    sorting_params = ss.get_default_sorter_params("kilosort4")
    # Run kilosort4 with defualt parameters
    sorting = ss.run_sorter(
        sorter_name="kilosort4", recording=raw_AP, output_folder=str(output_folder), **sorting_params
    )
    return


def get_ephys_paths_df():
    """
    Returns a pandas DataFrame with data extracted from raw ephys data folders.
    """
    return
