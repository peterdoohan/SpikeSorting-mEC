"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeinterface import extractors as se
from spikeinterface import sorters as ss
from spikeinterface import preprocessing as sp
from spikeinterface import widgets as sw
from spikeinterface import qualitymetrics as sq
from spikeinterface import exporters as sx
from kilosort import io
from kilosort import run_kilosort

from spikeinterface.core import load_extractor, create_sorting_analyzer

from probeinterface import ProbeGroup, write_prb

# %% Global variables
EPHYS_PATH = Path("../data/raw_data/ephys")
KILOSORT_PATH = Path("../data/preprocessed_data/Kilosort")
LFP_PATH = Path("../data/preprocessed_data/LFP")

ss.Kilosort3Sorter.set_kilosort3_path("./Kilosort3")

# ephys_path = Path('../data/raw_data/ephys/mEC_8/2024-02-22_18-08-35')
# AP_stream_name = 'Record Node 101#Neuropix-PXI-100.mEC8-AP'
#


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
def post_processing(preprocessed_path, kilosort_path):
    """ """
    preprocessed_recording = load_extractor(preprocessed_path)
    sorting = se.read_kilosort(kilosort_path)
    analyzer = create_sorting_analyzer(sorting, preprocessed_recording, sparse=True, format="memory", n_jobs=40)
    job_kwargs = dict(n_jobs=40, chunk_duration="1s", progress_bar=True)
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0, **job_kwargs)
    analyzer.compute("templates", operators=["average", "median", "std"])
    analyzer.compute("noise_levels")
    analyzer.compute("correlograms")
    analyzer.compute("unit_locations")
    analyzer.compute("spike_amplitudes", **job_kwargs)
    analyzer.compute("principal_components", **job_kwargs)
    analyzer.compute("spike_locations", **job_kwargs)
    metric_names = sq.get_quality_metric_list() + sq.get_quality_pca_metric_list()
    metrics = sq.compute_quality_metrics(analyzer, metric_names=metric_names)
    sx.export_report(analyzer, preprocessing_path.parent / "report", **job_kwargs)
    sx.export_to_phy()

    return


def spikesort_session2(ephys_path, AP_stream_name, run_kilosort4=True):
    subject, datetime_string = ephys_path.parts[-2:]
    output_folder = KILOSORT_PATH / subject / datetime_string
    output_folder.mkdir(parents=True, exist_ok=True)
    # preprocess ephys data with spikeinterface
    raw_AP = se.read_openephys(ephys_path, stream_name=AP_stream_name)
    all_channel_ids = raw_AP.get_channel_ids()
    phase_shift_AP = sp.phase_shift(raw_AP)
    # highpass_AP = sp.highpass_filter(phase_shift_AP, freq_min=300)
    _, channel_labels = sp.detect_bad_channels(phase_shift_AP)
    preprocessed_AP = phase_shift_AP.remove_channels(all_channel_ids[channel_labels == "out"])
    bad_channels = np.concatenate(
        [all_channel_ids[channel_labels == "noise"], all_channel_ids[channel_labels == "dead"]]
    )
    preprocessed_AP = sp.interpolate_bad_channels(preprocessed_AP, bad_channel_ids=bad_channels)
    preprocessed_AP = sp.highpass_spatial_filter(preprocessed_AP)
    preprocessed_AP.channel_ids
    # save preprocessed AP signal with spikeinterface
    print(f"Saving preprocessed data to {output_folder}")
    temp_output_folder = output_folder / "preprocessed_temp"
    preprocessed_AP.save(
        folder=temp_output_folder,
        format="binary",
        n_jobs=40,
        chunk_duration="1s",
        progress_bar=True,
        overwrite=True,
    )
    if run_kilosort4:
        run_kilosort4(preprocessed_AP, output_folder)
    if run_kilosort3:
        run_kilosort3(preprocessed_AP, output_folder)

    return


def run_kilosort4(preprocessed_AP, output_folder):
    """
    Runs kilosort4 without spikeinterface (due to bug in SI code that dosen't register GPU @20240605)
    Loads data preprocessed with spikeinterface and saves kilosort4 output to output folder / kilosort4.
    """
    n_channels = preprocessed_AP.channel_ids.shape[0]
    sample_freq = preprocessed_AP.sampling_frequency
    # save and load probe into KS
    probe = _load_kilosort_probe(preprocessed_AP, temp_output_folder)
    temp_output_folder = output_folder / "preprocessed_temp"
    raw_data_file = list(temp_output_folder.rglob("*.raw"))[0]
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort(
        settings={"fs": sample_freq, "n_chan_bin": n_channels},
        probe=probe,
        filename=raw_data_file,
        results_dir=output_folder / "kilosort4",
        data_dtype="int16",
        do_CAR=False,
    )
    return


def run_kilosort3(preprocessed_AP, output_folder):
    """
    Runs kilosort3 through spikeinterface. Note that this requires Kilosort3 to be installed and compiled locally,
    see README for more details."""
    sorter_params = ss.get_default_sorter_params("kilosort3")
    sorter_params["car"] = False
    sorter_params["do_correction"] = True
    ks3_sorting = ss.run_sorter(
        "kilosort3",
        recording=preprocessed_AP,
        folder=output_folder / "kilosort3",
        verbose=True,
        remove_existing_folder=True,
        **sorter_params,
    )
    return


def _load_kilosort_probe(si_recording, temp_output_folder):
    """
    This function saves a spikeinterface probe object to a .prb file that can be loaded
    into Kilosort4

    Inputs:
    - si_recording: spikeinterface recording object
    - temp_output_folder: Path to save the .prb file

    Returns:
    - KS probe dict
    """
    pg = ProbeGroup()
    pg.add_probe(probe=si_recording.get_probe())
    probe_path = temp_output_folder / "probe.prb"
    write_prb(probe_path, pg)
    return io.load_probe(probe_path)


# %%


def spikesort_session(
    ephys_path,
    AP_stream_name,
    interpolate_bad_channels=True,
    remove_bad_channels=False,
    denoise="destripe",
    save_temp_preprocessed_data=True,
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
    # motion correction
    # if motion_correction == "spikeinterface":
    #     motion_correction_folder = output_folder / "motion_correction"
    #     motion_correction_folder.mkdir(parents=True, exist_ok=True)
    #     motion_correction_AP = sp.correct_motion(
    #         preprocessed_AP, preset="nonrigid_accurate", folder=motion_correction_folder
    #     )
    if save_qc_figs:
        preprocessing_fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
        sw.plot_traces(raw_AP, backend="matplotlib", clim=(-500, 500), ax=axs[0])
        sw.plot_traces(preprocessed_AP, backend="matplotlib", clim=(-500, 500), ax=axs[1])
        for ax, label in zip(axs, ["Raw", "Preprocessed"]):
            ax.set_title(label)
            ax.set_xlabel("Time (s)")
        fig_folder = output_folder / "figs"
        fig_folder.mkdir(parents=True, exist_ok=True)
        preprocessing_fig.savefig(output_folder / "figs" / "preprocessed_traces.png")
    # Save preprocessed .dat file
    if save_temp_preprocessed_data:
        print(f"Saving preprocessed data to {output_folder}")
        preprocessed_AP.save(
            folder=output_folder / "preprocessed_temp",
            format="binary",
            n_jobs=40,
            chunk_duration="1s",
            progress_bar=True,
            overwrite=True,
        )
    # return preprocessed_AP
    # Run Kilosort4 with spikeinterface
    sorting_params = ss.get_default_sorter_params("kilosort4")
    sorting_params["do_CAR"] = False
    sorting_params["skip_kilosort_preprocessing"] = True
    sorting_params["do_correction"] = True
    print(f"Running Kilosort4 on {subject} {datetime_string}")
    sorting = ss.run_sorter(
        "kilosort4",
        recording=preprocessed_AP,
        output_folder=output_folder / "kilosort4",
        verbose=True,
        remove_existing_folder=True,
        **sorting_params,
    )
    return sorting


# %% Supporting Functions


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


# %% Plotting functions


def plot_kilosort_results(kilosort_dir, save_dir, save_name, version=4):
    """
    Adapted from https://github.com/MouseLand/Kilosort/blob/main/docs/tutorials/kilosort4.ipynb
    """
    # load kilosort output
    if version == 4:
        try:
            ops = np.load(kilosort_dir / "ops.npy", allow_pickle=True).item()
            camps = pd.read_csv(kilosort_dir / "cluster_Amplitude.tsv", sep="\t")["Amplitude"].values
            contam_pct = pd.read_csv(kilosort_dir / "cluster_ContamPct.tsv", sep="\t")["ContamPct"].values
            chan_map = np.load(kilosort_dir / "channel_map.npy")
            templates = np.load(kilosort_dir / "templates.npy")
            chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
            chan_best = chan_map[chan_best]
            amplitudes = np.load(kilosort_dir / "amplitudes.npy")
            spike_times = np.load(kilosort_dir / "spike_times.npy")
            clu = np.load(kilosort_dir / "spike_clusters.npy")
        except FileNotFoundError:
            raise FileNotFoundError(f"Kilosort output not found in directory: {kilosort_dir}")
    elif version == 3:
        try:
            camps = pd.read_csv(kilosort_dir / "cluster_Amplitude.tsv", sep="\t")["Amplitude"].values
            contam_pct = pd.read_csv(kilosort_dir / "cluster_ContamPct.tsv", sep="\t")["ContamPct"].values
            chan_map = np.load(kilosort_dir / "channel_map.npy")
            templates = np.load(kilosort_dir / "templates.npy")
            chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
            chan_best = chan_map[:, chan_best]
            amplitudes = np.load(kilosort_dir / "amplitudes.npy")
            spike_times = np.load(kilosort_dir / "spike_times.npy")
            clu = np.load(kilosort_dir / "spike_clusters.npy")
        except FileNotFoundError:
            raise FileNotFoundError(f"Kilosort output not found in directory: {kilosort_dir}")
    firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / spike_times.max()
    # plot figure
    gray = 0.5 * np.ones(3)
    fig = plt.figure(figsize=(10, 10), dpi=100)
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)
    # probe drift
    if version == 4:
        dshift = ops["dshift"]
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(np.arange(0, ops["Nbatches"]) * 2, dshift)
        ax.set_xlabel("time (sec.)")
        ax.set_ylabel("drift (um)")
    # probe raster
    ax = fig.add_subplot(grid[0, 1:])
    t0 = 0
    t1 = np.nonzero(spike_times > ops["fs"] * 5)[0][0]
    ax.scatter(spike_times[t0:t1] / 30000.0, chan_best[clu[t0:t1]], s=0.5, color="k", alpha=0.25)
    ax.set_xlim([0, 5])
    ax.set_ylim([chan_map.max(), 0])
    ax.set_xlabel("time (sec.)")
    ax.set_ylabel("channel")
    ax.set_title("spikes from units")
    # firing rate histogram
    ax = fig.add_subplot(grid[1, 0])
    nb = ax.hist(firing_rates, 20, color=gray)
    ax.set_xlabel("firing rate (Hz)")
    ax.set_ylabel("# of units")
    # amplitude histogram
    ax = fig.add_subplot(grid[1, 1])
    nb = ax.hist(camps, 20, color=gray)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("# of units")
    # contamination histogram
    ax = fig.add_subplot(grid[1, 2])
    nb = ax.hist(np.minimum(100, contam_pct), np.arange(0, 105, 5), color=gray)
    ax.plot([10, 10], [0, nb[0].max()], "k--")
    ax.set_xlabel("% contamination")
    ax.set_ylabel("# of units")
    ax.set_title("< 10% = good units")

    for k in range(2):
        ax = fig.add_subplot(grid[2, k])
        is_ref = contam_pct < 20.0
        ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color="r", label="mua", alpha=0.25)
        ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color="b", label="good", alpha=0.25)
        ax.set_ylabel("amplitude (a.u.)")
        ax.set_xlabel("firing rate (Hz)")
        ax.legend()
        if k == 1:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("loglog")
    fig.savefig(save_dir / f"{save_name}.png")
    return
