"""A script to unitmatch for tracking cells across sessions (within and across days).
Primarily based off of UMPy_spike_interface_demo.ipynb.
Integrated with other scripts to spikesort from raw data collected using open_ephys.
Notably, inherits paths from spikesort_session.
@charlesdgburns"""

import os
from pathlib import Path
from datetime import date


from . import spikesort_session as sps
from . import run_ephys_preprocessing as run_e

#import UnitMatchPy
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import UnitMatchPy.save_utils as su
import UnitMatchPy.GUI as gui
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params


## SET UP UNIT_MATCH PARAMETERS

PARAM = default_params.get_default_param() #default is ready for Neuropixel 2.0 probes

PARAM['no_shanks'] = 1 #Neuropixel 1.0 probe
PARAM['shank_dist'] = 0 #Neuropixel 1.0 probe
#Changing distances and radii might be useful for across-day matching.
PARAM['max_dist'] = 250 #default for Neuropixel 2.0 is 100
PARAM['channel_radius'] = 400 #default for Neuropixel 2.0 is 150

def run_unit_match():
    ''' Top level function to submit a bunch of jobs for unit matching.'''
    #Find all within-subject session pairs where preproecssing has been completed
    
    
def unit_match_sessions(subject = str, datetime1 = str, datetime2 = str, params = PARAM):
    '''Code to unit_match sessions, taking inputs in the form of:
    INPUT: subject and datetime isoformat strings. These need to match data path formats.
    OUTPUT: [...]
    '''
    #Setting up paths
    all_session_paths = [(sps.SPIKESORTING_PATH/subject/datetime1/'UM_inputs'),
                        (sps.SPIKESORTING_PATH/subject/datetime2/'UM_inputs')]
    UM_out_path = Path('../data/preprocessed_data/UnitMatch')/subject/f'{datetime1}x{datetime2}'
    UM_out_path.mkdir(parents=True,exist_ok=True)
    param['session_paths'] = all_session_paths
    
   
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(all_session_paths)
    print('reading Raw waveform data')
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths, unit_label_paths, param, 
                                                                                                       good_units_only = True) #this can break if = False; unit match doesn't detect bad units itself.

    #param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

    # create clus_info, contains all unit id/session related info
    clus_info = { 'session_switch' : session_switch, 
                'session_id' : session_id, 
                'original_ids' : np.concatenate(good_units) }

    #Extract parameters from waveform into a wave properties dictionary
    wave_dict = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    #Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors  = ov.extract_metric_scores(wave_dict, session_switch, within_session, param, niter  = 2)

    #Probability analysis
    output_prob_matrix = get_output_prob_matrix(param,total_score, candidate_pairs, scores_to_include, predictors)
    
    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > 0.75] = 1 #might want to set match threshold to something other than 0.75.
    
    matches = [] #empty list here since we're not curating using GUI.
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)
    su.save_to_output(UM_out_path, scores_to_include, matches, output_prob_matrix, 
                    wave_dict['avg_centroid'], wave_dict['avg_waveform'], wave_dict['avg_waveform_per_tp'], wave_dict['max_site'],
                   total_score, output_threshold, clus_info, param, UIDs = UIDs, matches_curated = None, save_match_table = True)
    #save out extracted_wave_properties 
    return print(f'Saved to: {UM_out_path}')

def get_unitmatch_reports(subject = str, datetime1 = str, datetime2 = str):
    UM_out_path = Path('../data/preprocessed_data/UnitMatch')/subject/f'{datetime1}x{datetime2}'


## Utility

def get_output_prob_matrix(param, total_score, candidate_pairs, scores_to_include, predictors):
    prior_match = 1 - (param['n_expected_matches'] / param['n_units']**2 ) # freedom of choose in prior prob
    priors = np.array((prior_match, 1-prior_match))

    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = param['score_vector']
    parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one = 1)

    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

    output_prob_matrix = probability[:,1].reshape(param['n_units'],param['n_units'])
    return output_prob_matrix

def pad_UM_inputs(preprocessed_path):
    ''' Function to account for different numbers of channels between sessions,
    due to outside brain channels being removed in preprocessing.'''
    max_n_channels = 384 #set to neuropixel 1.0 count

    # First we pad positions, shaped [n_channels,n_coords]:
    positions = np.load(preprocessed_path / 'UM_inputs' / 'channel_positions.npy')
    if positions.shape[0]<max_n_channels:
        n_pad = max_n_channels-positions.shape[0]
        positions_padded = np.concatenate([positions, np.zeros((n_pad, 2))], axis=0)
        print(f'Padding data to {max_n_channels} channels')
        np.save((preprocessed_path / 'UM_inputs' / 'channel_positions.npy'), positions_padded)
    
    # Next, we pad units
    raw_waveforms_dir = preprocessed_path / 'UM_inputs' / 'RawWaveforms'
    unit_files = list(raw_waveforms_dir.glob('Unit*_RawSpikes.npy'))
    for unit_file in unit_files:
        # Load raw spikes and positions for each unit
        raw_spikes = np.load(unit_file)

        # double-check padding is needed
        if raw_spikes.shape[1]<max_n_channels:
            # Pad raw spikes along the second axis (channels)
            n_pad = max_n_channels - raw_spikes.shape[1]
            raw_spikes_padded = np.concatenate([raw_spikes, np.zeros((raw_spikes.shape[0], n_pad, raw_spikes.shape[2]))], axis=1)
            # Save the padded data, overwriting the original files
            np.save(unit_file, raw_spikes_padded)  # Overwrite the raw spikes file 
    return

    
def zero_center_waveform(waveform):
    """
    Centers waveform about zero, by subtracting the mean of the first 15 time points.
    This function is useful for Spike Interface where the waveforms are not centered about 0.

    Arguments:
        waveform - ndarray (nUnits, Time Points, Channels, CV)

    Returns:
        Zero centered waveform
    """
    waveform = waveform -  np.broadcast_to(waveform[:,:15,:,:].mean(axis=1)[:, np.newaxis,:,:], waveform.shape)
    return waveform

## Development // debugging

def send_test_jobs(subject = 'mEC_2', date_str='2024-02-20'):
    '''Sends jobs for an individual subject and date for kilosort preprocessing.
    Also saves out unit match inputs'''
    ephys_df = sps.get_ephys_paths_df()
    ephys_df['date'] = ephys_df['datetime'].apply(lambda x: x.date())
    subject_df = ephys_df[ephys_df['subject_ID']==subject]
    sessions_df = subject_df[subject_df['date']==date.fromisoformat(date_str)]
    for each_session in range(len(sessions_df)): #hardcoding here is a bit ugly, but go find a few within day sessions.
        ephys_info = sessions_df.iloc[each_session]   
        if ephys_info['spike_interface_readable'] == True:
            run_e.submit_test_job(ephys_info)
    return print('Submitted a few jobs to test preprocessing')

def test_unit_match():
    ephys_df = sps.get_ephys_paths_df()
    ephys_df = ephys_df.sort_values(by=['datetime'])
    sessions_paths =  []
    for index in [19,20]:
        ephys_info = ephys_df.iloc[index]
        sessions_paths.append(ephys_info['ephys_path'])

def hack_cluster_group(preprocessed_path):
    '''A bit of a hack to get around UM issue for all units.'''

def save_cluster_group(preprocessed_path, hack=False):
    read_path = preprocessed_path/'kilosort4'/'sorter_output'/'cluster_group.tsv'
    save_path = preprocessed_path/'UM_inputs'/'cluster_group.tsv'
    cluster_group_df = pd.read_csv(read_path,sep='\t')
    if hack == True:
            cluster_group_df['KSLabel'] = 'good'
    cluster_group_df.to_csv(save_path, sep='\t', index=False)
    
    return print('Copied over cluster group')