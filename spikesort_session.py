"""Script to spikesort ephys sessions using Kilosort and Spike interface"""

# %%  imports
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
import matplotlib.pyplot as plt
from spikeinterface import extractors as se
from spikeinterface import sorters as ss
from spikeinterface import preprocessing as sp
from spikeinterface import widgets as sw
from spikeinterface import qualitymetrics as sq
from spikeinterface import exporters as sx
from spikeinterface import curation as sc
from kilosort import io
from kilosort import run_kilosort

from spikeinterface.core import load_extractor, create_sorting_analyzer

from probeinterface import ProbeGroup, write_prb

# %% Global variables
EPHYS_PATH = Path("../data/raw_data/ephys")
KILOSORT_PATH = Path("../data/preprocessed_data/Kilosort")

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


def preprocess_ephys_session(
    ephys_path,
    AP_stream_name,
    save_preprocessed_data=True,
    load_preprocessed_data=False,
    remove_processed_data=False,
    spike_sorter="Kilosort3",
    remove_duplicate_spikes=False,
    print_summary=True,
):
    """
    This function takes a data from raw ephys Neuropixel 1.0 recordings aquired with OpenEphys and creates a preprocessed
    data folder that contains temporary preprocessed_data file (optionally deleted at the end of preprocessing), the outputs
    of spikesorting (Kilosort3 or Kilosort4), and report that gives detailed quality metrics and summary figures for each
    spike sorted unit.

    The function runs the following steps:
        EPHYS_PREPROCESSING:
        - Phase shift corrrection (accounts for time between sampling neuropixel data across channels)
        - Highpass filter (300Hz)
        - Detect and remove channels outside the brain
        - Detect and interpolate bad channels
        - Denoise IBL highpass_spatial_filter method (destrip protocol)
        - Save out raw and preprocessed traces for quality control (needed for fast spikesorting)
        - Save out binary preprocessed data /preprocessed_temp

        SPIKESORTING:
        - Run Kilosort3 or Kilosort4 without common average referencing (drift correction is applied)
        - Removes excess spikes (removes spikes attributed to outside recording period)
        - (Optionally) remove duplicate spikes (just trying this out, not sure it is a good idea)

        POSTPROCESSING:
        - Comute quality metrics for spike sorted units
        - Save out report with quality metrics and summary figures
        - Optionaly print cluster ids that pass IBL single-unit quality control
    """
    # set up output folder
    subject, datetime_string = ephys_path.parts[-2:]
    output_folder = KILOSORT_PATH / subject / datetime_string
    output_folder.mkdir(parents=True, exist_ok=True)
    # preprocessing
    raw_rec = se.read_openephys(ephys_path, stream_name=AP_stream_name)
    temp_preprocessed_dir = output_folder / "preprocessed_temp"
    if not load_preprocessed_data:
        preprocessed_rec = preprocess_ephys_data(raw_rec, save_dir=temp_preprocessed_dir)
    else:
        preprocessed_rec = load_extractor(temp_preprocessed_dir)
    # save preprocssed traces for quality control
    fig = _plot_preprocessed_trace_qc(raw_rec, preprocessed_rec)
    fig.savefig(output_folder / "preprocessed_traces.png")
    # spikesorting
    Kilosort_dir = output_folder / spike_sorter
    if spike_sorter == "Kilosort4":
        assert save_preprocessed_data or load_preprocessed_data, "Kilosort4 requires preprocessed data to be saved"
        sorter = run_Kilosort4(preprocessed_rec, temp_preprocessed_dir, Kilosort_dir)
    elif spike_sorter == "Kilosort3":
        sorter = run_Kilosort3(preprocessed_rec, Kilosort_dir)
    else:
        raise ValueError(f"{spike_sorter} not supported")
    sorter = sc.remove_excess_spikes(sorter, preprocessed_rec)
    sorter = sc.remove_duplicated_spikes(sorter) if remove_duplicate_spikes else sorter
    # postprocessing & quality metics
    qc_report_path = Kilosort_dir / "report"
    postprocess_ephys_data(preprocessed_rec, sorter, qc_report_path)
    qc_metric_df = pd.read_csv(qc_report_path / "quality metrics.csv")
    qc_pass_single_units = _get_qc_pass_single_units(qc_metric_df)
    if remove_processed_data:
        temp_output_folder = output_folder / "preprocessed_temp"
        shutil.rmtree(temp_output_folder)
    if print_summary:
        print(f"Total clusters fround: {len(qc_metric_df)}")
        print(f"Clusters passing sing unit QC: Total: {len(qc_pass_single_units)}, {qc_pass_single_units}")
    return print(f"Finished processing ephys session: {subject} {datetime_string}")


def preprocess_ephys_data(raw_rec, save_dir):
    """Preprocesses ephys data with spikeinterface, see preprocess_ephys_session for details"""
    all_channel_ids = raw_rec.get_channel_ids()
    phase_shift_rec = sp.phase_shift(raw_rec)
    highpass_rec = sp.highpass_filter(phase_shift_rec, freq_min=300)
    _, channel_labels = sp.detect_bad_channels(phase_shift_rec)
    preprocessed_rec = highpass_rec.remove_channels(all_channel_ids[channel_labels == "out"])
    bad_channels = np.concatenate(
        [all_channel_ids[channel_labels == "noise"], all_channel_ids[channel_labels == "dead"]]
    )
    preprocessed_rec = sp.interpolate_bad_channels(preprocessed_rec, bad_channel_ids=bad_channels)
    preprocessed_AP = sp.highpass_spatial_filter(preprocessed_rec)
    # save preprocessed AP signal with spikeinterface
    if save_dir:
        print(f"Saving preprocessed ephys data to {save_dir}")
        preprocessed_AP.save(
            folder=save_dir,
            format="binary",
            n_jobs=80,
            chunk_duration="2s",
            progress_bar=True,
            overwrite=True,
        )
    return preprocessed_AP


def _plot_preprocessed_trace_qc(raw_rec, preprocessed_rec):
    """Saves figure with raw and preprocessed traces for quality control"""
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
    sw.plot_traces(raw_rec, backend="matplotlib", clim=(-100, 100), ax=axs[0], return_scaled=True)
    sw.plot_traces(preprocessed_rec, backend="matplotlib", clim=(-100, 100), ax=axs[1], return_scaled=True)
    for ax, label in zip(axs, ["Raw", "Preprocessed"]):
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
    return fig


def run_Kilosort4(preprocessed_rec, temp_preprocessed_dir, Kilosort_dir):
    """Run kilosort4 without spikeinterface (current bug in SI code, adapt when fixed)"""
    n_channels = preprocessed_rec.channel_ids.shape[0]
    sample_freq = preprocessed_rec.sampling_frequency
    # save and load probe into KS
    probe = _load_Kilosort4_probe(preprocessed_rec, temp_preprocessed_dir)
    raw_data_file = list(temp_preprocessed_dir.rglob("*.raw"))[0]
    sorter_output_dir = Kilosort_dir / "sorter_output"  # match output when using spikeinterface
    sorter_output_dir.mkdir(parents=True, exist_ok=True)
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort(
        settings={"fs": sample_freq, "n_chan_bin": n_channels},
        probe=probe,
        filename=raw_data_file,
        results_dir=sorter_output_dir,
        data_dtype="int16",
        do_CAR=False,
    )
    sorter = se.read_kilosort(sorter_output_dir)
    return sorter


def _load_Kilosort4_probe(si_recording, temp_output_folder):
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


def run_Kilosort3(preprocessed_AP, Kilosort_dir):
    """
    Runs kilosort3 through spikeinterface. Note that this requires Kilosort3 to be installed and compiled locally,
    see README for more details."""
    sorter_params = ss.get_default_sorter_params("kilosort3")
    sorter_params["car"] = False
    sorter_params["do_correction"] = True
    sorter = ss.run_sorter(
        "kilosort3",
        recording=preprocessed_AP,
        folder=Kilosort_dir,
        verbose=True,
        remove_existing_folder=True,
        **sorter_params,
    )
    return sorter


def postprocess_ephys_data(preprocessed_rec, sorter, report_path):
    """Computes quality metrics and saves report for spike sorted data"""
    analyzer = create_sorting_analyzer(sorter, preprocessed_rec, sparse=True, format="memory", n_jobs=40)
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
    # save report
    sx.export_report(analyzer, report_path, **job_kwargs)
    return


def _get_qc_pass_single_units(
    qc_metrics_df, sliding_rp_violation_range=(0.1, 0.9), amplitude_cutoff_thres=0.05, median_aplitude_thres=50
):
    """
    Filters quality metrics output dataframe for units that pass single unit quality control (similar to how IBL does it).
    See https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html for metric descriptions.
    Args:
        report_path (Path): Path to the Kilosort report folder
        sliding_rp_violation_range (tuple): Tuple of floats, range of rp violation values that are acceptable
        amplitude_cutoff_thres (float): Float, amplitude cutoff threshold
        median_aplitude_thres (float): Float, median amplitude threshold
    """
    qc_metrics_df.reset_index(inplace=True)
    qc_metrics_df.rename(columns={"index": "unit_id"}, inplace=True)
    qc_metrics_df["amplitude_median"] = qc_metrics_df["amplitude_median"].abs()
    single_units_qc_passed = qc_metrics_df.query(
        f"sliding_rp_violation > {sliding_rp_violation_range[0]} and sliding_rp_violation < {sliding_rp_violation_range[1]} and amplitude_cutoff < {amplitude_cutoff_thres} and amplitude_median > {median_aplitude_thres}"
    )
    return single_units_qc_passed["unit_id"].to_list()


# %%


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
