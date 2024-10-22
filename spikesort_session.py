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

#for preprocessing and kilosort
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
# for saving unit match inputs:
import UnitMatchPy.extract_raw_data as erd





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
    subject_ID, datetime, ephys_path, kilosort_Ths=[9,8], 
    save_QC_plots=True, cache_preprocessed_data=True, remove_cached_data=True,
    spikesort_path = SPIKESORTING_PATH
):
    """NB: kilosort_Ths = [Th_universal, Th_learned] for optimising over different parameters."""
    # Set up filepaths
    preprocessed_path = Path(spikesort_path) / subject_ID / datetime
    temp_path = preprocessed_path / "temp_preprocessed"

    print(f'Checking {preprocessed_path} for DONE.txt')

    if not (preprocessed_path/"DONE.txt").exists():
        if not temp_path.exists(): #If a temporary cached file already exists, we can go straight to loading (see else:)
            print('Not DONE... preprocessing data')
            preprocessed_path.mkdir(parents=True, exist_ok=True)
            raw_rec = se.read_openephys(ephys_path, stream_id="0")  # stream_id="0" is the AP data
            preprocessed_rec = denoise_ephys_data(raw_rec, preprocessed_path, plot=save_QC_plots)
            if cache_preprocessed_data:
                print("Caching preprocessed data")
                if not temp_path.exists():
                    temp_path.mkdir()
                preprocessed_rec = preprocessed_rec.save(
                    folder=preprocessed_path / "temp_preprocessed",
                    format="binary",
                    overwrite=True,
                )
        else:
            print("loading cached preprocessed data")
            preprocessed_rec = si.load_extractor(preprocessed_path / "temp_preprocessed")
        print("running spikesorting")
        sorter = run_kilosort4(preprocessed_rec, preprocessed_path, kilosort_Ths)
        print('Computing quality metrics...')
        quality_metrics_df = get_quality_metrics(sorter, preprocessed_rec, preprocessed_path, save_cluster_reports=True)
        quality_metrics_df.to_csv(preprocessed_path / "quality_metrics.htsv", sep="\t", index=False)
        single_units = get_single_units(quality_metrics_df)
        print(f"Found {len(single_units)} single units, passing quality control")
        print('Saving out waveforms for UnitMatch.')
        save_unitmatch_inputs(preprocessed_rec, preprocessed_path)
        # Remove temp files
        if remove_cached_data:
            for temp_folder in ["temp_preprocessed", "sorting_analyser"]:
                temp_path = preprocessed_path / temp_folder
                if temp_path.exists():
                    shutil.rmtree(temp_path)
        open(preprocessed_path/"DONE.txt",'w').close() #stores an empty .txt file to check for completion.
    return print(f"completed ephys preprocessing for {subject_ID} {datetime}")


def denoise_ephys_data(raw_rec, preprocessed_path, plot=True):
    # phase shift correcting 
    raw_rec = check_rec_properties(raw_rec) #check for neuropixel properties; this was missing for some subjects
    phase_shift_rec = sp.phase_shift(raw_rec)
    # high pass filter
    highpass_rec = sp.highpass_filter(phase_shift_rec)
    # remove channels outside brain and interpolate bad channels
    channel_assignments = get_channel_assignments(highpass_rec, preprocessed_path)
    outside_brain_channels = [k for k, v in channel_assignments.items() if v == "out"]
    if len(outside_brain_channels) > 0:
        preprocessed_rec = highpass_rec.remove_channels(outside_brain_channels)
    else:
        outside_brain_channels = None
        preprocessed_rec = highpass_rec
    bad_channels = [k for k, v in channel_assignments.items() if v in ["noise", "dead",]]
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
    highpass_rec, preprocessed_path, #always required
    outside_thres=0.3, n_neighbours=37, save_params=False, #for saving parameters
    plot_outside_brain_channels=True, #option to avoid plotting
):
    """
    Finds channels outside the brain ('out'), noisey channels ('noise'), and good channels ('good') from raw ephys
    recording and saves this information to disk as a .json file. 
    This should be run for each session, after checking parameter selection for a each subject.
    See notebooks/preprocessing_ephys.ipynb for more detail.
    """
    # Set up paths as global directory (particularly for optim_kilosort functionality)
    preprocessed_path=Path(preprocessed_path) #make sure this is a path object not 'str'
    subject_id = preprocessed_path.parts[-2]
    params_dir = SPIKESORTING_PATH/subject_id/'channel_assign_params.json' #this is a global directory
    
    #1) if we're saving params, generate new stuff with input parameters.
    #2) if we're not saving params, generate new stuff with loaded parameters
  
    if save_params:
        params = {'outside_threshold':outside_thres,
                  'n_neighbours':n_neighbours}
        with open(params_dir, "w") as outfile:
            outfile.write(json.dumps(params, indent=4))
    else:
        try:
            with open(params_dir, 'r') as j:
                params = json.loads(j.read())
        except:
            raise print(f'Failed loading from {params_dir} \n Must verify parameters for bad channel assignment. Please see preprocessing_ephys.ipynb')
    print('Assigning bad channels...')
    #Now we've got our parameters, we will generate new channel_assignments and save them.
    all_channel_ids = highpass_rec.get_channel_ids()
    _, channel_labels = sp.detect_bad_channels(
        highpass_rec,
        outside_channels_location="top",
        outside_channel_threshold=params['outside_threshold'],
        noisy_channel_threshold = 1, #default value but specified  
        dead_channel_threshold = -0.5, #default value but specified
        n_neighbors = params['n_neighbours'], # above we choose 37 as twice the size (due to nyquist) of a missing chunk in mEC_5.
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
        if preprocessed_path is None:
            raise ValueError("need specific preprocessed path to plot")
        else:
                    
            fig, axs = plt.subplots(ncols=2)

            mask = np.tile(channel_labels!='good',(10,1))
            axs[0].imshow(mask.T, aspect='auto',origin='lower')        
            axs[0].set(title='Bad channels mask')

            sw.plot_traces(highpass_rec, 
                        channel_ids=all_channel_ids,
                        backend="matplotlib",
                        clim=(-100, 100),
                        ax=axs[1],
                        return_scaled=True,
                        time_range=[500, 550], #arbitrary 1ms time window ~8 min into rec
                        order_channel_by_depth=True,
                        show_channel_ids=False,)
            axs[1].set(title='Highpass data', xlabel='time (100ms)')
            fig.savefig(preprocessed_path/ "outside_brain_channels.png")
    # channel assignments dict
    channel_id2assignment = {all_channel_ids[i]: channel_labels[i] for i in range(len(all_channel_ids))}
    with open(preprocessed_path / "channel_assignments.json", "w") as outfile:
        outfile.write(json.dumps(channel_id2assignment, indent=4))

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


def run_kilosort4(preprocessed_rec, preprocessed_path, kilosort_Ths=[9,8], IBL_preprocessing = True):
    """ Runs kilosort4 after preprocessing using spike-interface.
    We allow changes to Th_universal and Th_learned for optimisation, leaving all other parameters default.
    We also allow a toggle for IBL-style preprocessing of data, noting that Kilosort processes raw data faster."""
    kilosort_output_path = preprocessed_path / "kilosort4"
    #load best Th parameters if kilosort parameters have been optimised. Otherwise default is given above.
    if (SPIKESORTING_PATH/'kilosort_optim'/'best_params.json').exists(): 
        with open(SPIKESORTING_PATH/'kilosort_optim'/'best_params.json', 'r') as f:
            kilosort_Ths = json.load(f)
    if not (preprocessed_path/'kilosort4').exists(): #if the ks folder exists, assume sorting completed with no bugs.
        print("running Kilosort4")
        kilosort_output_path.mkdir(parents=True)
        sorter_params = ss.get_default_sorter_params("kilosort4")
        sorter_params["do_CAR"] = False #we perform IBL destriping instead using spikeinterface
        sorter_params["Th_universal"] = kilosort_Ths[0]
        sorter_params["Th_learned"] = kilosort_Ths[1]
        sorter_params["nblocks"] = 5 #Default is 1 (rigid), 5 is recommended for single shank neuropixel. Shouldn't have a big influence regardless.
        sorter = ss.run_sorter(
            "kilosort4",
            recording=preprocessed_rec,
            folder=kilosort_output_path,
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


def save_unitmatch_inputs(preprocessed_rec, preprocessed_path):
    '''To be run within spikesort_session.py after kilosort outputs but before temp_processed folder is deleted.
    INPUTS: path/to/preprocessed/spikesorting/sub/session, 
            IBL preprocessed data in spikeinterface format,
            path/to/preprocessed/sub/spikesorting/session/kilosort4/sorter_output,
    OUTPUTS: raw waveforms for a session for later unit matching. 
             path/to/preprocessed/spikesorting/sub/session/UM_inputs
'''
    
    #First we set up our recording and sorting
    recording = preprocessed_rec
    sorting = se.read_kilosort(preprocessed_path/'kilosort4'/'sorter_output')
    #get a list of units used - we use all of them
    unit_used = sorting.get_property('original_cluster_id')

    #Split our data into two, so we can get averaged waveforms
    split_idx = recording.get_num_samples() // 2
    #for the sorting
    split_sorting = []
    split_sorting.append(sorting.frame_slice(start_frame=0, end_frame=split_idx))
    split_sorting.append(sorting.frame_slice(start_frame=split_idx, end_frame=recording.get_num_samples()))
    #for the recording
    split_recording = []
    split_recording.append(recording.frame_slice(start_frame=0, end_frame=split_idx))
    split_recording.append(recording.frame_slice(start_frame=split_idx, end_frame=recording.get_num_samples()))

    #Next, we create a sorting analyzer to get the average waveform
    split_analysers = []
    split_analysers.append(si.create_sorting_analyzer(split_sorting[0], split_recording[0], sparse=False))
    split_analysers.append(si.create_sorting_analyzer(split_sorting[1], split_recording[1], sparse=False))
    
    for half in range(2):
        split_analysers[half].compute(
            "random_spikes",
            method="uniform",
            max_spikes_per_unit=500)
        split_analysers[half].compute('templates', n_jobs = 0.8)
    templates_first = split_analysers[0].get_extension('templates')
    templates_second = split_analysers[1].get_extension('templates')
    t1 = templates_first.get_data()
    t2 = templates_second.get_data()
    all_waveforms = np.stack((t1,t2), axis = -1) #here we store all the waveforms

    #That's all we need! now we save it all!
    UM_input_dir = preprocessed_path/'UM_inputs'
    UM_input_dir.mkdir(exist_ok=True,parents=True)
    erd.save_avg_waveforms(all_waveforms, UM_input_dir.resolve(), #absolute path for unitmatch 
                                good_units=0, extract_good_units_only = False)
    positions = split_analysers[0].get_channel_locations() #save positions from one of the split halves
    np.save(UM_input_dir/'channel_positions.npy', positions)
    save_unitmatch_labels(preprocessed_path) #Separate funciton as we might have to run and adjust later.
    
    return print(f'UnitMatch waveforms saved to {UM_input_dir}')
    
def save_unitmatch_labels(preprocessed_path):
    ''' INPUT: path object to preprocessed data
        OUTPUT: cluster_group.tsv file with 'mua' and 'good' labels assigned by quality metrics'''
    quality_metrics_df = pd.read_csv(preprocessed_path/'cluster_reports'/'quality metrics.csv')
    #assign single units with lenient thresholding
    single_units = get_single_units(quality_metrics_df,
    isi_violations_ratio_thres=0.2, #0.2 instead of 0.1
    amplitude_cutoff_thres=0.1,
    firing_rate_thres=0.1,
    presence_ratio_thres=0.8, #0.8 instead of 0.9
    amplitude_median_thres=40) #40 instead of 50
    cluster_group_df = pd.read_csv(preprocessed_path/'kilosort4'/'sorter_output'/'cluster_group.tsv', sep='\t')
    cluster_group_df['KSLabel'] = 'mua'
    cluster_group_df.loc[single_units, 'KSLabel'] ='good'
    cluster_group_df.to_csv(preprocessed_path/'UM_inputs'/'cluster_group.tsv', 
                            sep='\t', index=False) #index false is important here.
    return cluster_group_df

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
    #two conditions where we find errors:
    if not (SPIKESORTING_PATH/"probe_params").exists():
        save_rec_properties()

    true_location = np.load(SPIKESORTING_PATH/'probe_params'/'location.npy')
    
    if not (raw_rec.get_property('location')==true_location).all():
        #Properties about the probe are missing or have errors, so we are adding them now:
        for property in ['contact_vector','location','group','inter_sample_shift']:
            property_array = np.load(SPIKESORTING_PATH/"probe_params"/f'{property}.npy')
            raw_rec.set_property(property,property_array)
       
    return raw_rec
        
def save_rec_properties():
    ''' Function to save out properties of the probe, simply based on the first recording.
    NB: these might need to be double-checked, since we had a few probes where the location property was weird.
    See function below.'''  
    index = 0 # <-- this might have to be changed if the first recording doesn't work.
    raw_rec = se.read_openephys(get_ephys_paths_df().iloc[index].ephys_path, 
                            stream_id="0")
    if raw_rec.get_property('inter_sample_shift') is not None: ## Using a recording /with/ the properties.
        print('Saving out properties of probe.')
        (SPIKESORTING_PATH/'probe_params').mkdir(exist_ok=True)
        for property in ['contact_vector','location','group','inter_sample_shift']:
            np.save(SPIKESORTING_PATH/"probe_params"/property,raw_rec.get_property(property))
    else: #If the first recording by chance doesn't have the correct properties, try another.
        raise print('Failed saving out probe properties. See save_rec_properties() function.')

def get_loc_err():
    '''Debugging code to identify errors in open_ephys recordings.
    some recordings were missing properties or had wrong location properties.
    This adds a column to ephys_paths_df which identifies such cases.'''
    ephys_paths_df = get_ephys_paths_df()

    property_error = []
    for each_session in ephys_paths_df.ephys_path:
        try:
            raw_rec = se.read_openephys(each_session, stream_id="0")
            checked_location = location = np.load(SPIKESORTING_PATH/"probe_params"/'location.npy')
            if raw_rec.get_property('inter_sample_shift') is None:
                property_error.append('missing')
            elif (raw_rec.get_property('location') == checked_location).all():
                property_error.append('true')
            else:
                property_error.append('wrong')
                print(raw_rec.get_property('location'))
        except:
            property_error.append('not_readable')

    ephys_paths_df['loc_err'] = property_error
    return ephys_paths_df            
# %% Filepath management


def get_ephys_paths_df():
    """Tries to open all raw ephys files to check that they are readable (this can take time).
        This is stored as tsv file and later read and updated to check whether processing is completed."""
    all_ephys_paths = [f for s in EPHYS_PATH.iterdir() if s.is_dir() for f in s.iterdir() if f.is_dir()]
    all_spikesorting_paths = [f for s in SPIKESORTING_PATH.iterdir() if s.is_dir() for f in s.iterdir() if f.is_dir()]
    
    if (EPHYS_PATH/"ephys_paths_df.tsv").exists(): #load the df if it already exists.
        ephys_paths_df = pd.read_csv(EPHYS_PATH/"ephys_paths_df.tsv", sep='\t')
        ephys_paths_df['datetime']=ephys_paths_df['datetime'].apply(lambda x: pd.to_datetime(x)) #change datetime from str to object
        #update completion
        completion = []
        for path in all_ephys_paths:
            subject_ID = path.parts[-2]
            datetime_string = path.parts[-1]
            dt = datetime.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
            spike_sorting_completed =(SPIKESORTING_PATH/subject_ID/dt.isoformat()/"DONE.txt").exists()
            completion.append(spike_sorting_completed)
        ephys_paths_df['spike_sorting_completed'] = completion
    else: #otherwise generate dataframe from scratch
        ephys_path_info = [] 
        for path in all_ephys_paths:
            subject_ID = path.parts[-2]
            datetime_string = path.parts[-1]
            dt = datetime.strptime(datetime_string, "%Y-%m-%d_%H-%M-%S")
            spike_sorting_completed = (SPIKESORTING_PATH/subject_ID/dt.isoformat()/'DONE.txt').exists()
            try:
                rec = se.read_openephys(path, stream_id="0")
                duration_min = rec.get_num_frames() / (30000*60)
                spike_interface_readable = True
            except:
                spike_interface_readable = False

            ephys_path_info.append(
                {
                    "subject_ID": subject_ID,
                    "datetime": dt,
                    "ephys_path": str(path),
                    "spike_sorting_completed": spike_sorting_completed,
                    "spike_interface_readable": spike_interface_readable,
                    "duration_min": duration_min,
                }
                )
        ephys_paths_df = pd.DataFrame(ephys_path_info)

    #save for future readout
    ephys_paths_df.to_csv(EPHYS_PATH/"ephys_paths_df.tsv", sep = '\t', index=False)
    
    return ephys_paths_df

# %% Running kilosort on raw data



# %% tests
def run_test():
    ephys_paths_df = get_ephys_paths_df()
    ephys_info = ephys_paths_df.iloc[0]
    # preprocessing
    preprocessed_path = SPIKESORTING_PATH / ephys_info.subject_ID / ephys_info.datetime.isoformat()
    if not (preprocessed_path/'temp_preprocessed').exists():
        preprocessed_path.mkdir(parents=True)
        raw_rec = se.read_openephys(ephys_info.ephys_path, stream_id="0")  # stream_id="0" is the AP data
        preprocessed_rec = denoise_ephys_data(raw_rec, preprocessed_path, plot=True)
        print("Caching preprocessed data")
        preprocessed_rec = preprocessed_rec.save(
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


