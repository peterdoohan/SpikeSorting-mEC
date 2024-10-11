"""A script to unitmatch for tracking cells across sessions (within and across days).
Primarily based off of UMPy_spike_interface_demo.ipynb.
Integrated with other scripts to spikesort from raw data collected using open_ephys
@charlesdgburns"""

import os
from pathlib import Path
from datetime import date#

import UnitMatchPy
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import numpy as np
import matplotlib.pyplot as plt
import UnitMatchPy.save_utils as su
import UnitMatchPy.GUI as gui
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
import UnitMatchPy.extract_raw_data as erd

from . import spikesort_session as sps
from . import run_ephys_preprocessing as run_e


## 

def save_unitmatch_inputs(preprocessed_path, preprocessed_data):
    '''To be run within spikesort_session.py after kilosort outputs but before temp_processed folder is deleted.
    INPUTS: path/to/preprocessed/spikesorting/sub/session, 
            IBL preprocessed data in spikeinterface format,
            path/to/preprocessed/sub/spikesorting/session/kilosort4/sorter_output
    OUTPUTS: raw waveforms for a session for later unit matching. 
             path/to/preprocessed/spikesorting/sub/session/UM_inputs
'''
    
    #First we set up our recording and sorting
    recording = preprocessed_data
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
    split_analysers.append(si.create_sorting_analyzer(split_sorting[1], recording[1], sparse=False))
    all_waveforms = [] #here we store all the waveforms
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
    all_waveforms.append(np.stack((t1,t2), axis = -1))
    #at this point we need a channel positions array
    all_positions = []
    for i in range(2):
        #positions for first half and second half are the same
        all_positions.append(split_analysers[i].get_channel_locations())
    
    #That's all we need! now we save it all!
    UM_input_dir = preprocessed_path/'UM_inputs'
    UM_input_dir.mkdir(exist_ok=True)
    for i in range(2): #for each split half
        erd.save_avg_waveforms(all_waveforms[i], UM_input_dir, good_units=0, extract_good_units_only = False)
        np.save(UM_input_dir/'channel_positions.npy', all_positions[i])
    return print(f'UnitMatch waveforms saved to {UM_input_dir}')
    
    
def send_test_jobs():
    '''Sends mEC2 first day sessions to be processed.'''
    ephys_df = sps.get_ephys_paths_df()
    ephys_df = ephys_df.sort_values(by=['datetime'])
    for index in [17,19,20]: #hardcoding here is a bit ugly, but go find a few within day sessions.
        ephys_info = ephys_df.iloc[index]
        run_e.submit_test_job(ephys_info)
    return print('Submitted a few jobs to test UM outputs')
