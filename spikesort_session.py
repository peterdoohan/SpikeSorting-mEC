"""
Integrated code to

1) pre-process ephys data with IBL pipeline,
2) Run KS4 and 
3) compute quality metrics for ephys data

This is at 'session' level so we aim to support different probe types as well as multiple probes per subject/session.

@peterdoohan and @charlesdgburns
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

#get probe data for cambridge neurotech probes
from probeinterface.plotting import plot_probe
from probeinterface import Probe, get_probe

# for saving unit match inputs:
import UnitMatchPy.extract_raw_data as erd

# %% MANUAL SETUP / INPUTS 

SAMPLING_FREQUENCY = 30000 #30kHz sampling rate used for AP data. This is used to check that the correct data is loaded.
N_PROBES = 1 #Support to handle multiple probes. See also function get_probe_recordings():

# %% Global Variables 
# Paths should not be changed if data is structured as expected and if current directory is path/to/experiment/code
# This is input to load ephys_data:
EPHYS_PATH = Path("../data/raw_data/ephys") # Path("/Volumes/behrens/peter_doohan/goalNav_opto-mFC/ephys_control/data/raw_data/ephys")
# This is output of spike sorted data:
SPIKESORTING_PATH = Path("../data/preprocessed_data/spikesorting")  # Path("/Volumes/behrens/peter_doohan/goalNav_opto-mFC/ephys_control/data/preprocessed_data/spikesorting")


si.set_global_job_kwargs(n_jobs=80, chunk_duration="1s", progress_bar=True)
# %% Top Level Function
def preprocess_ephys_session(subject_ID, datetime, ephys_path, #Required inputs 
    IBL_preprocessing=True, kilosort_Ths=[9,8], #Options for preprocessing and kilosort parameters
    save_QC_plots=True, cache_preprocessed_data=True, remove_cached_data=True, #QC and caching options
    spikesort_path = SPIKESORTING_PATH):
    
    """ Top level function to process all the ephys data related to a session.
    Here we perform the following steps: 
    0. separate probes if required
    1. preprocess data with IBL standards (per probe shank) and cache data for spikeinterface
    2. run kilosort (on each probe) and save outputs
    3. save inputs for UnitMatch (across sessions)
    4. get quality metrics
    5. delete temporarily cached data
    
    NB: kilosort_Ths = [Th_universal, Th_learned] for optimising over different parameters."""
    
    #0.  check recording probe properties and get output filepaths (handles separate probes if required):
    raw_recs, preprocessed_paths = get_probe_recordings(subject_ID, datetime, ephys_path, spikesort_path)
    for raw_rec, preprocessed_path in zip(raw_recs, preprocessed_paths):
        temp_path = preprocessed_path / "temp_preprocessed"
        print(f'Checking {preprocessed_path} for DONE.txt')    
        if not (preprocessed_path/"DONE.txt").exists():
            if not temp_path.exists(): #If a temporary cached file already exists, we can go straight to loading (see else:)
                print('Not DONE... preprocessing data')
                preprocessed_path.mkdir(parents=True, exist_ok=True)        
                stream_id = get_stream_id(ephys_path, data_type='AP')
                raw_rec = se.read_openephys(ephys_path, stream_id=stream_id)  # stream_id="0" is the AP data
                preprocessed_rec = pre_kilosort_processing(raw_rec, preprocessed_path, IBL_preprocessing, plot=save_QC_plots)
                    
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
        sorter = run_kilosort4(preprocessed_rec, preprocessed_path, kilosort_Ths, IBL_preprocessing)
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

# %% _1. IBL preprocessing functions
def pre_kilosort_processing(raw_rec, preprocessed_path, IBL_preprocessing = True, plot=True):
    
    # Automatic per_shank processing and later merging
    n_shanks = len(np.unique(raw_rec.get_property('group')))
    split_recordings = raw_rec.split_by(property='group')
    preprocessed_recs = []
    for each_shank in range(n_shanks):
        recording = split_recordings[each_shank]

        # phase shift correcting 
        if len(raw_rec.get_property('inter_sample_shift'))!=0:
            shank_rec = sp.phase_shift(raw_rec) #we make changes to this 'shank_rec' object at shank level
        
        if IBL_preprocessing == True:
            # high pass filter
            shank_rec = sp.highpass_filter(shank_rec)
            # remove channels outside brain and interpolate bad channels
            channel_assignments = get_channel_assignments(shank_rec, preprocessed_path)
            outside_brain_channels = [k for k, v in channel_assignments.items() if v == "out"]
            if len(outside_brain_channels) > 0:
                recording = recording.remove_channels(outside_brain_channels)
            else:
                outside_brain_channels = None
                #do nothing
            bad_channels = [k for k, v in channel_assignments.items() if v in ["noise", "dead",]]
            if len(bad_channels) > 0:
                shank_rec = sp.interpolate_bad_channels(shank_rec, bad_channel_ids=bad_channels)
            # destripe (denoising)
            n_channels_on_shank = len(shank_rec.get_property('group'))
            n_channel_pad=min(60,n_channels_on_shank)
            shank_rec = sp.highpass_spatial_filter(shank_rec, n_channel_pad=n_channel_pad)
            preprocessed_recs.append(shank_rec)
        else:
            preprocessed_recs.append(shank_rec) #append only 
        
        preprocessed_rec = si.aggregate_channels(preprocessed_recs)
    
        
        # plot traces for visual inspection
        if plot:
            print("saving preprocessed traces (for each shank) for visual inspection")
            fig = _plot_preprocessed_trace_qc(split_recordings, preprocessed_recs)
            fig.savefig(preprocessed_path / f"preprocessed_trace_qc_shank_{each_shank}.png")
    return preprocessed_rec


def get_channel_assignments(highpass_rec, preprocessed_path, #Required inputs 
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
    params_dir = SPIKESORTING_PATH/'probe_params'/subject_id/'channel_assign_params.json' #this is a global directory
    
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
                        time_range=[500, 505], #arbitrary 1ms time window ~8 min into rec
                        order_channel_by_depth=True,
                        show_channel_ids=False,)
            axs[1].set(title='Highpass data', xlabel='time (100ms)')
            fig.savefig(preprocessed_path/ "outside_brain_channels.png")
    # channel assignments dict
    channel_id2assignment = {all_channel_ids[i]: channel_labels[i] for i in range(len(all_channel_ids))}
    with open(preprocessed_path / "channel_assignments.json", "w") as outfile:
        outfile.write(json.dumps(channel_id2assignment, indent=4))

    return channel_id2assignment


def _plot_preprocessed_trace_qc(split_recordings, preprocessed_recs):
    """Saves figure with raw and preprocessed traces for quality control.
    This is plotted per shank"""
    n_shanks = len(preprocessed_recs)
    fig, axs = plt.subplots(2,n_shanks, figsize=(n_shanks*10, 40))
    for each_shank in range(n_shanks):
        
        #sort out axes indexing
        if n_shanks == 1:
            top_axis = axs[0]
            bottom_axis = axs[1]
        else:
            top_axis = axs[0,each_shank]
            bottom_axis = axs[1,each_shank]

        top_axis.set(title = f'Raw shank {each_shank+1}', xlabel='Time(s)')
        sw.plot_traces(
            split_recordings[each_shank],
            backend="matplotlib",
            clim=(-100, 100),
            ax=top_axis,
            return_scaled=True,
            time_range=[0, 5],
            order_channel_by_depth=True,
            show_channel_ids=True,
        )

        bottom_axis.set(title = f'preprocessed shank {each_shank+1}', xlabel='Time(s)')
        sw.plot_traces(
            preprocessed_recs[each_shank],
            backend="matplotlib",
            clim=(-100, 100),
            ax=bottom_axis,
            return_scaled=True,
            time_range=[0, 5],
            order_channel_by_depth=True,
            show_channel_ids=True,
        )
    return fig

# $$ _2. Kilosort via spikeinterface
def run_kilosort4(preprocessed_rec, preprocessed_path, kilosort_Ths=[9,8]):
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

# %% _3. Spike sorting quality control via spikeinterface
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
    sd_ratio_thres=3,
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
    remaining_query = f"amplitude_cutoff < {amplitude_cutoff_thres} and firing_rate > {firing_rate_thres} and presence_ratio > {presence_ratio_thres} and amplitude_median > {amplitude_median_thres} and sd_ratio < {sd_ratio_thres}"
    qc_pass_df = qc_pass_df.query(remaining_query)
    return qc_pass_df.unit_id.values

# %% _4. Saving unitmatch inputs from cached data

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


# %% _0. Handling probe and recording information

def get_probe_recordings(subject_ID, datetime, ephys_path, 
                         n_probes = N_PROBES, spikesort_path = SPIKESORTING_PATH):
    '''INPUT: ephys_path to raw ephys data of a session
       OUTPUT: list of recordings and list of preprocessed_paths 
       
       Notes: At this stage we also pass functions which check that probe information is loaded properly.'''
    recordings = []
    preprocessed_paths = []

    if n_probes == 1: #start with simple case
        preprocessed_paths.append(Path(spikesort_path) / subject_ID / datetime)
        stream_id = get_stream_id(ephys_path, data_type='AP')
        raw_rec = se.read_openephys(ephys_path, stream_id=stream_id)  # stream_id="0" is assumed to be AP data
        raw_rec = check_rec_probe(raw_rec, subject_ID)
        recordings.append(raw_rec)
    
    elif n_probes == 2: #dual probe setup
        # at this stage assume different probes are named probe_A, probe_B.
        # We assume there's a single recording but two probes, which we split here.
        # This can be adjusted if there are multiple files per session.
        raw_rec = get_stream_id(ephys_path, data_type='AP')
        raw_rec = se.read_openephys(ephys_path, stream_id=stream_id)
        
        for i,probe in enumerate(['probe_A','probe_B']): #renaming probes could happen at preprocessed -> processed data stage.
            preprocessed_paths.append(Path(spikesort_path) / subject_ID / datetime/ f'{probe}')
            
            #TODO: may need to split recordings in a different way
            # split by total number of channels
            total_channels = raw_rec.get_num_channels()
            half_channels = total_channels // 2
            channel_indexing = np.add(list(range(half_channels)),half_channels*i) #first or second half for A or B.
            channel_ids = raw_rec.channel_ids[channel_indexing]
            raw_rec = raw_rec.channel_slice(channel_ids)
            raw_rec = check_rec_probe(raw_rec, subject_ID)
            
            recordings.append(raw_rec)
    return recordings, preprocessed_paths

def get_stream_id(ephys_path, data_type='AP'):
    '''INPUT: ephys_path and specification to read AP or LFP data.
       OUTPUT: the correct stream_id for spike-interface open ephys reading.
       
       Notes: fixes an issue where multiple node recordings duplicated data in subject folders.
       multiple stream id's would misalign to the incorrect subject data.
       We do this instead of removing data; this data check relies on probes being named by subject,
       and channels being named by datatype (AP or LFP). 
       
       Defaults to reading the first stream_id=0, i.e., reading the first folder in
       '../data/raw_data/ephys/subject/datetime/Record Node X/experiment1/recording1/continuous/'''
       
    #Strategy here is to iterate over stream_id's until we get a matching subject and datatype.
    #If there's no match, we default to stream_id = "0", 
    # explicitly, we assume that there's only one subject's data and it's sampled at 30kHz.
    data_type = 'AP'
    subject_str = Path(ephys_path).parts[-2].replace("_","") #open_ephys doesn't allow underscores.
    try:
        for i in range(8):
            rec = se.read_openephys(ephys_path, stream_id=f'{i}')
            subject_match = (subject_str == rec.get_annotation('probes_info')[0]['name'])
            data_match = True if f'{data_type}' in rec.channel_ids[0] else False
            if (subject_match and data_match):
                stream_id = i #successful if probe name and datatype match.
                break
            else: 
                stream_id=0
    except: #if there's no matches or the data is not readable:
        stream_id=0

    return str(stream_id) #must be string format.

def check_rec_probe(raw_rec, subject_ID, probe_suffix = None):
    '''Function to double-check that the properties of the probe are saved with it.
        These are properties related to the probes used, but can be missing in some recordings.
        
        probe_suffix argument to be passed through when there is more than one probe.'''
    
    #We want to make sure locations match with saved probe locations,
    #but we also want to avoid overwriting probe information when unnecessary
    
    #1. load saved probe
    probe_params_path = Path(SPIKESORTING_PATH/'probe_params'/subject_ID)
    if probe_suffix is not None:
        probe_params_path = probe_params_path/probe_suffix
    saved_probe = pd.read_csv(probe_params_path/'probe_layout.tsv')

    #2. compare to probe on recording (if available - otherwise just load saved probe)
    try:
        probe_df = raw_rec.get_probe() #check that a probe can be returned
        print('Successfully extracted probe from recording.')
        locations_from_rec = raw_rec.get_property('location')
        order = np.argsort(saved_probe['device_channel_indices'])
        locations_from_saved_probe = saved_probe[['x','y']].to_numpy()[order]
        if np.all(locations_from_rec==locations_from_saved_probe)!=True:
            raise print('Locations not matching saved probe.')
        
    except: 
        print('Loading probe from saved data.')
        raw_rec = raw_rec.set_probe(Probe.from_dataframe(saved_probe), group_mode='by_shank')

    return raw_rec
        
def save_rec_probe(subject_ID:str, #required
                        manufacturer = None, probe_name = None, wiring_device = None, 
                        probe_suffix = None, manual_ephys_path = None):
    ''' Function to save out properties of the probe at each subject level.
    -For neuropixel probes, we find the first recording with the data saved and use for future fixes.
    -For cambridgeneurotech probes we rely on probeinterface and MANUAL inputs.
    INPUT EXAMPLES:
        manufacturer =  'cambridgeneurotech'
        probe_name = 'ASSY-236-F'
        wiring_device = 'cambridgeneurotech_mini-amp-64'
        probe_suffix = 'probe_A' #NB: for multi-probes only, as probe info is saved in subject subfolders
    OUTPUT: preprocessed_data/probe_params/subject/probe_layout.csv 
    
    NB: these should be double-checked, since we had a few probes where the location property was weird.
    see also get_loc_error()''' 
    
    #If recording neuropixel data, it suffices to find a single recording with the correct file,
    # this then gets saved to fix failed files later with check_rec_probe().

    #set up save path 
    probe_params_path = SPIKESORTING_PATH / 'probe_params'/ subject_ID #save path for probe parameters
    if probe_suffix is not None:
        probe_params_path = probe_params_path/probe_suffix
    
    #choose 
    index = 0 # <-- this might have to be changed if the first recording doesn't work.
    ephys_path = get_ephys_paths_df().query('subject_ID==subject_ID').loc[index,'ephys_path']
    if manual_ephys_path is not None:
        ephys_path = manual_ephys_path
    stream_id = get_stream_id(ephys_path)
    raw_rec = se.read_openephys(ephys_path, stream_id=stream_id)
    try:
        probe = raw_rec.get_probe()
        print('Succesfully retrieved probe from recording')
    except:
        print('Failed to retrieve probe from recording \n OBS! Manually saving via probeinterface')
        probe = get_probe(manufacturer=manufacturer, #probeinterface function
                          probe_name = probe_name)
        probe.wiring_to_device(wiring_device) #this wiring may be wrong. Check probe documentation.
    probe_df = probe.to_dataframe()
    probe_df['device_channel_indices'] = probe.device_channel_indices
    probe_params_path.mkdir(parents=True,exist_ok=True)
    probe_df.to_csv(probe_params_path/'probe_layout.tsv', index=False)
    #also save out visual check
    fig, ax = plt.subplots()
    plot_probe(probe, ax=ax) #probeinterface function
    fig.savefig(probe_params_path/'probe_plot.png')
    
    return print(f'Saved out probe data to \n {probe_params_path}')

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
       
 # %% Filepath management - this is quite fundamental to all top level functions.

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
                stream_id = get_stream_id(path)
                rec = se.read_openephys(path, stream_id=stream_id)
                if rec.sampling_frequency != SAMPLING_FREQUENCY:
                    raise print(f'Data sampled at {rec.sampling_frequency}, not matching expected frequency.')
                duration_min = rec.get_num_frames() / (SAMPLING_FREQUENCY*60)
                spike_interface_readable = True
            except:
                spike_interface_readable = False
                duration_min = 0

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

# %% TESTING!


#FOR DEBUGGING:
## Debugging with test_paths:

#beatriz dual probe cambridge neurotech data:
#ephys_path = '/ceph/behrens/Beatriz/beatriz/7x7_Maze_HC_Rec_FEB2023/03_recordings_data/MR34/2023-02-21_15-11-11' 
#francesca neuropixel 2.0 data:
#ephys_path = '/ceph/behrens/Francesca/compReplay_mEC/experiment/data/raw_data/ephys/MR2_NM/2024-06-22_12-54-16'
#peter/charles neuropixel 1.0 data:
#ephys_path = '/ceph/behrens/peter_doohan/goalNav_mEC/experiment/data/raw_data/ephys/mEC_5/2024-02-18_12-15-42'
#peter cambridge neurotech data:
#ephys_path = '/ceph/behrens/peter_doohan/goalNav_mFC/experiment/data/raw_data/ephys/m2/2022-06-22_14-05-23'


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


