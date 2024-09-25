"""
Run KS4 and compute quality metrics for ephys data
@peterdoohan
"""

# %% Imports
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from spikeinterface import core as si
from spikeinterface import extractors as se
from spikeinterface import preprocessing as sp
from spikeinterface import widgets as sw
from spikeinterface import sorters as ss
from spikeinterface import curation as sc
from spikeinterface import postprocessing as pp
from spikeinterface import qualitymetrics as qm
from spikeinterface import exporters as sx
from spikeinterface import qualitymetrics as sq


# %% Global Variables
EPHYS_PATH = Path(
    "../data/raw_data/ephys"
)  # Path("/Volumes/behrens/peter_doohan/goalNav_opto-mFC/ephys_control/data/raw_data/ephys")
SPIKESORTING_PATH = Path(
    "../data/preprocessed_data/spikesorting"
)  # Path("/Volumes/behrens/peter_doohan/goalNav_opto-mFC/ephys_control/data/preprocessed_data/spikesorting")


si.set_global_job_kwargs(n_jobs=80, chunk_duration="1s", progress_bar=True)
# %% Functions


def preprocess_ephys_session(
    subject_ID, datetime, ephys_path, save_QC_plots=True, cache_preprocessed_data=True, remove_cached_data=True
):
    """ """
    # Set up filepaths
    preprocessed_path = SPIKESORTING_PATH / subject_ID / datetime
    temp_path = preprocessed_path / "temp_preprocessed"
    if not preprocessed_path.exists():
        preprocessed_path.mkdir(parents=True)
        raw_rec = se.read_openephys(ephys_path, stream_id="0")  # stream_id="0" is the AP data
        preprocessed_rec = denoise_ephys_data(raw_rec, preprocessed_path, plot=save_QC_plots)
        if cache_preprocessed_data:
            print("Caching preprocessed data")
            if not temp_path.exists():
                temp_path.mkdir()
            preprocessed_rec.save(
                folder=preprocessed_path / "temp_preprocessed",
                format="binary",
                overwrite=True,
            )
    else:
        print("loading cached preprocessed data")
        preprocessed_rec = si.load_extractor(preprocessed_path / "temp_preprocessed")
    # spikesorting
    sorter = run_kilosort4(preprocessed_rec, preprocessed_path)
    # Compute quality metrics
    quality_metrics_df = get_quality_metrics(sorter, preprocessed_rec, preprocessed_path, save_cluster_reports=True)
    quality_metrics_df.to_csv(preprocessed_path / "quality_metrics.htsv", sep="\t", index=False)
    single_units = get_single_units(quality_metrics_df)
    print(f"Found {len(single_units)} single units, passing quality control")
    # Remove temp files
    if remove_cached_data:
        for temp_folder in ["temp_preprocessed", "sorting_analyser"]:
            temp_path = preprocessed_path / temp_folder
            if temp_path.exists():
                shutil.rmtree(temp_path)
    return print(f"completed ephys preprocessing for {subject_ID} {datetime}")


def denoise_ephys_data(raw_rec, preprocessed_path, plot=True):
    # phase shift correcting 
    raw_rec = check_rec_properties(raw_rec) #check for neuropixel properties; this was missing for some subjects
    phase_shift_rec = sp.phase_shift(raw_rec)
    # high pass filter
    highpass_rec = sp.highpass_filter(phase_shift_rec)
    # remove channels outside brain and interpolate bad channels
    channel_assignments_path = preprocessed_path.parent / "channel_assignments.json"
    if not channel_assignments_path.exists():
        raise print(
            f"Subject {preprocessed_path.parts[-2]} has no channel assignments. \n Run get_channel_assignments; see README or preprocessing_ephys notbook"
        )
    else:
        with open(channel_assignments_path) as infile:
            channel_assignments = json.load(infile)
    outside_brain_channels = [k for k, v in channel_assignments.items() if v == "out"]
    if len(outside_brain_channels) > 0:
        preprocessed_rec = highpass_rec.remove_channels(outside_brain_channels)
    else:
        outside_brain_channels = None
        preprocessed_rec = highpass_rec
    bad_channels = [k for k, v in channel_assignments.items() if v in ["noise", "dead"]]
    if len(bad_channels) > 0:
        preprocessed_rec = sp.interpolate_bad_channels(preprocessed_rec, bad_channel_ids=bad_channels)
    # destripe (denoising)
    preprocessed_rec = sp.highpass_spatial_filter(preprocessed_rec)
    # plot traces for visual inspection
    if plot:
        print("saving preprocessed traces for visual inspection")
        fig = _plot_preprocessed_trace_qc(raw_rec, preprocessed_rec)
        fig.savefig(preprocessed_path / f"preprocessed_trace_qc.png")
    return preprocessed_rec


def get_channel_assignments(
    highpass_rec, outside_thres=0.3, #NB spikeinterface default is -0.5. 
    plot_outside_brain_channels=True, subject_ID=None, save=False
):
    """
    Finds channels outside the brain ('out'), noisey channels ('noise'), and good channels ('good') from raw ephys
    recording and saves this information to disk as a .json file. This should be run once per subject and used for all
    of that subject's sessions during further preprocessing.
    """
    all_channel_ids = highpass_rec.get_channel_ids()
    _, channel_labels = sp.detect_bad_channels(
        highpass_rec,
        outside_channels_location="top",
        outside_channel_threshold=outside_thres,
        seed=0,
    )
    outside_brain_channels = all_channel_ids[channel_labels == "out"] if (channel_labels == "out").sum() > 0 else None
    noisey_channels = all_channel_ids[channel_labels == "noise"] if (channel_labels == "noise").sum() > 0 else None
    dead_channels = all_channel_ids[channel_labels == "dead"] if (channel_labels == "dead").sum() > 0 else None
    for label, channels in zip(
        ["outside_brain", "noisey", "dead"], [outside_brain_channels, noisey_channels, dead_channels]
    ):
        if channels is None:
            print(f"No {label} channels found")
        else:
            print(f"{len(channels)} {label} channel(s) found")
    if plot_outside_brain_channels:
        if subject_ID is None:
            raise ValueError("need specific subject ID to plot")
        else:
            fig, axs = plt.subplots(ncols=2, figsize=(40, 60))
            sw.plot_traces(
                highpass_rec,
                channel_ids=all_channel_ids[channel_labels != "out"],
                backend="matplotlib",
                clim=(-100, 100),
                ax=axs[0],
                return_scaled=True,
                time_range=[500, 505], #arbitrary time window ~8 min into rec
                order_channel_by_depth=True,
                show_channel_ids=True,
            )
            if not outside_brain_channels is None:
                sw.plot_traces(
                    highpass_rec,
                    channel_ids=outside_brain_channels,
                    backend="matplotlib",
                    clim=(-100, 100),
                    ax=axs[1],
                    return_scaled=True,
                    time_range=[500, 505], #arbitrary time window ~8 min into rec
                    order_channel_by_depth=True,
                    show_channel_ids=True,
                )
            else:
                print("no outside brain channels... nothing to plot")
            fig.savefig(SPIKESORTING_PATH / subject_ID / "outside_brain_channels.png")
    # channel assignments dict
    channel_id2assignment = {all_channel_ids[i]: channel_labels[i] for i in range(len(all_channel_ids))}
    if save:
        if subject_ID is None:
            raise print("need subject_ID specified to plot")
        else:
            with open(SPIKESORTING_PATH / subject_ID / "channel_assignments.json", "w") as outfile:
                outfile.write(json.dumps(channel_id2assignment, indent=4))
            print(f"saved channel assignments to {SPIKESORTING_PATH / subject_ID}")
    else:
        return channel_id2assignment


def _plot_preprocessed_trace_qc(raw_rec, preprocessed_rec):
    """Saves figure with raw and preprocessed traces for quality control"""
    fig, axs = plt.subplots(ncols=2, figsize=(40, 60))
    sw.plot_traces(
        raw_rec,
        backend="matplotlib",
        clim=(-100, 100),
        ax=axs[0],
        return_scaled=True,
        time_range=[0, 5],
        order_channel_by_depth=True,
        show_channel_ids=True,
    )
    sw.plot_traces(
        preprocessed_rec,
        backend="matplotlib",
        clim=(-100, 100),
        ax=axs[1],
        return_scaled=True,
        time_range=[0, 5],
        order_channel_by_depth=True,
        show_channel_ids=True,
    )
    for ax, label in zip(axs, ["Raw", "Preprocessed", "Outside Brain"]):
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
    return fig


def run_kilosort4(preprocessed_rec, preprocessed_path):
    """ """
    kilosort_output_path = preprocessed_path / "kilosort4"
    if not kilosort_output_path.exists():
        print("running Kilosort4")
        kilosort_output_path.mkdir(parents=True)
        sorter_params = ss.get_default_sorter_params("kilosort4")
        sorter_params["skip_kilosort_preprocessing"] = True
        sorter = ss.run_sorter(
            "kilosort4",
            recording=preprocessed_rec,
            output_folder=kilosort_output_path,
            verbose=True,
            remove_existing_folder=True,
            **sorter_params,
        )
        sorter = sc.remove_excess_spikes(sorter, preprocessed_rec)
        sorter = sorter.remove_empty_units()
    else:  # if ks already run load sorter
        print("loading Kilosort4 output")
        sorter = ss.read_sorter_folder(
            kilosort_output_path,
            register_recording=preprocessed_rec,
        )
    return sorter


def get_quality_metrics(sorter, preprocessed_rec, preprocessed_path, save_cluster_reports):
    sorter = sc.remove_excess_spikes(sorter, preprocessed_rec)
    analyzer_output_path = preprocessed_path / "sorting_analyzer"
    if analyzer_output_path.exists():
        print("Loading Analyzer from Disk")
        analyzer = si.load_sorting_analyzer(analyzer_output_path)
    else:
        analyzer = si.create_sorting_analyzer(
            sorter,
            preprocessed_rec,
            sparse=True,
            format="binary_folder",
            folder=analyzer_output_path,
            overwrite=True,
        )
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
        analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0)
        print("calculating templates")
        analyzer.compute("templates", operators=["average", "median", "std"])
        print("calculating noise levels")
        analyzer.compute("noise_levels")
        print("calculating correlograms")
        analyzer.compute("correlograms")
        print("calculating unit locations")
        analyzer.compute("unit_locations")
        print("calculating spike amplitudes")
        analyzer.compute("spike_amplitudes")
        print("calculating spike locations")
        analyzer.compute("spike_locations")
        print("calculating template similarity")
        analyzer.compute("template_similarity")
    print("computing quality metrics df")
    metrics_list = sq.get_quality_metric_list()
    metrics_df = sq.compute_quality_metrics(analyzer, metric_names=metrics_list)
    # process metrics df
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "unit_id"}, inplace=True)
    metrics_df["amplitude_median"] = metrics_df["amplitude_median"].abs()
    # save cluster reports
    if save_cluster_reports:
        print("exporting cluster reports")
        try:
            sx.export_report(
                analyzer,
                output_folder=preprocessed_path / "cluster_reports",
                remove_if_exists=True,
            )
        except ValueError:
            print("Failed generating cluster report")
    return metrics_df


def get_single_units(
    quality_metric_df,
    isi_violations_ratio_thres=0.1,
    amplitude_cutoff_thres=0.1,
    firing_rate_thres=0.1,
    presence_ratio_thres=0.9,
    amplitude_median_thres=50,
):
    """
    Filter sortered clusters by quality metrics to find single units (clusters that pass QC metrics)

    Metrics:
    """
    isi_violations_mask = np.logical_or.reduce(
        [
            quality_metric_df.isi_violations_ratio < isi_violations_ratio_thres,
        ]
    )
    qc_pass_df = quality_metric_df[isi_violations_mask]
    remaining_query = f"amplitude_cutoff < {amplitude_cutoff_thres} and firing_rate > {firing_rate_thres} and presence_ratio > {presence_ratio_thres} and amplitude_median > {amplitude_median_thres}"
    qc_pass_df = qc_pass_df.query(remaining_query)
    return qc_pass_df.unit_id.values

def check_rec_properties(raw_rec):
    '''Function to double-check that the properties of the probe are saved with it.
        These are properties related to the probes used, but can be missing in some recordings.'''
    if raw_rec.get_property('inter_sample_shift') is None:
        #Properties about the probe are missing, so we are adding them now:
        for property in ['contact_vector','location','group','inter_sample_shift']:
            if not (SPIKESORTING_PATH/"params").exists():
                print('Missing properties. See check_rec_properties function.')
            property_array = np.load(SPIKESORTING_PATH/"params"/f'{property}.npy')
            raw_rec.set_property(property,property_array)

        #NB: if the property files are missing, we originally stored them as follows:
        #if raw_rec.get_property('inter_sample_shift') is not None: ## Using a recording /with/ the properties.
        #    for property in ['contact_vector','location','group','inter_sample_shift']:
        #        sps.np.save(sps.SPIKESORTING_PATH/"params"/property,raw_rec.get_property(property))
    return raw_rec
        
            
# %% Filepath management


def get_ephys_paths_df():
    """Tries to open all raw ephys files to check that they are readable (this can take time)"""
    all_ephys_paths = [f for s in EPHYS_PATH.iterdir() if s.is_dir() for f in s.iterdir() if f.is_dir()]
    all_spikesorting_paths = [f for s in SPIKESORTING_PATH.iterdir() if s.is_dir() for f in s.iterdir() if f.is_dir()]
    ephys_path_info = []
    for path in all_ephys_paths:
        subject_ID = path.parts[-2]
        datetime_string = path.parts[-1]
        dt = datetime.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
        spike_sorting_completed = (
            True if SPIKESORTING_PATH / subject_ID / dt.isoformat() in all_spikesorting_paths else False
        )
        try:
            se.read_openephys(path, stream_id="0")
            spike_interface_readable = True
        except:
            spike_interface_readable = False
        ephys_path_info.append(
            {
                "subject_ID": subject_ID,
                "datetime": dt.isoformat(),
                "ephys_path": str(path),
                "spike_sorting_completed": spike_sorting_completed,
                "spike_interface_readable": spike_interface_readable,
            }
        )
    return pd.DataFrame(ephys_path_info)


# %% tests
def run_test():
    ephys_paths_df = get_ephys_paths_df()
    ephys_info = ephys_paths_df.iloc[0]
    # preprocessing
    preprocessed_path = SPIKESORTING_PATH / ephys_info.subject_ID / ephys_info.datetime
    if not preprocessed_path.exists():
        preprocessed_path.mkdir(parents=True)
        raw_rec = se.read_openephys(ephys_info.ephys_path, stream_id="0")  # stream_id="0" is the AP data
        preprocessed_rec = denoise_ephys_data(raw_rec, preprocessed_path, plot=True)
        print("Caching preprocessed data")
        preprocessed_rec.save(
            folder=preprocessed_path / "temp_preprocessed",
            format="binary",
            overwrite=True,
        )
    else:
        print("loading cached preprocessed data")
        preprocessed_rec = si.load_extractor(preprocessed_path / "temp_preprocessed")
    # spikesorting
    sorter = run_kilosort4(preprocessed_rec, preprocessed_path)
    print("computing quality metrics")
    quality_metrics_df = get_quality_metrics(
        preprocessed_rec, sorter, preprocessed_path, save_cluster_reports=False
    )  # cluster reports currently bugged in spikeinterface
    quality_metrics_df.to_csv(preprocessed_path / "quality_metrics.htsv", sep="\t", index=False)
    single_units = get_single_units(quality_metrics_df)
    print(f"Found {len(single_units)} single units, passing quality control")
    return
