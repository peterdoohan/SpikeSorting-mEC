"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
from pathlib import Path
from datetime import datetime as dt
from spikeinterface import extractors as se
from spikeinterface import sorters as ss

# %% Global variables
KILOSORT_FOLDER = "../data/raw_data/Kilosort"

TEST_SESSIONS = [
    "../data/raw_data/ephys/mEC_6/2024-02-20_12-55-26",
    "../data/raw_data/ephys/mEC_6/2024-02-20_14-22-35",
    "../data/raw_data/ephys/mEC_6/2024-02-24_12-42-47",
    "../data/raw_data/ephys/mEC_6/2024-02-24_14-14-18",
    "../data/raw_data/ephys/mEC_6/2024-02-27_12-13-53",
    "../data/raw_data/ephys/mEC_6/2024-02-27_13-46-14",
    "../data/raw_data/ephys/mEC_6/2024-03-09_14-34-06",
    "../data/raw_data/ephys/mEC_6/2024-03-09_16-07-05",
    "../data/raw_data/ephys/mEC_6/2024-03-17_11-30-10",
    "../data/raw_data/ephys/mEC_6/2024-03-17_12-57-33",
    "../data/raw_data/ephys/mEC_6/2024-03-18_12-23-37",
    "../data/raw_data/ephys/mEC_6/2024-03-18_14-15-04",
]

TROUBLESHOOT_SESSION = "../data/raw_data/ephys/mEC_6/2024-02-20_12-55-26/Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-100.mEC6-AP/continuous.dat"

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
