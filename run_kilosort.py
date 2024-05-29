"""This script runs Kilosort4 on a given ephys session"""

# %% Imports
import numpy as np
from pathlib import Path
from datetime import datetime as dt


# %% Global variables
KILOSORT_PATH = "../data/raw_data/Kilosort"
#

ephys_paths = [
    "../data/raw_data/ephys/mEC_6/2024-02-27_13-46-14",  # Done :)
    "../data/raw_data/ephys/mEC_6/2024-03-09_14-34-06",  # Done :)
    "../data/raw_data/ephys/mEC_6/2024-03-09_16-07-05",
    "../data/raw_data/ephys/mEC_6/2024-03-17_11-30-10",
    "../data/raw_data/ephys/mEC_6/2024-03-17_12-57-33",
    "../data/raw_data/ephys/mEC_6/2024-03-18_12-23-37",
]

TROUBLESHOOT_SESSION = "../data/raw_data/ephys/mEC_6/2024-02-20_12-55-26/Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-100.mEC6-AP/continuous.dat"


# %% Functions
# def run_kilosort4(ephys_path, specified_output_folder=False):
#     # define output folder:
#     subject, collection_datetime = Path(ephys_path).parts[-2:]
#     sortname = specified_output_folder if specified_output_folder else dt.now().isoformat()
#     output_folder = Path(KILOSORT_PATH) / subject / collection_datetime / sortname
#     output_folder.mkdir(parents=True, exist_ok=True)
#     # Load data
#     raw_AP = se.read_openephys(ephys_path, stream_id="0")  # stream_id 0 is AP data, 1 is LFP data
#     filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
#         raw_AP,
#         output_folder,
#         data_name="data.bin",
#         dtype=np.int16,
#         chunksize=60000,
#         export_probe=True,
#         probe_name="probe.prb",
#     )
#     settings = {"fs": fs, "n_chan_bin": c}
#     probe = io.load_probe(probe_path)
#     ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort(
#         settings=settings,
#         probe=probe,
#         filename=filename,
#         results_dir=output_folder,
#     )
#     return
