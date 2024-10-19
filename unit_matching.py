"""A script to unitmatch for tracking cells across sessions (within and across days).
Primarily based off of UMPy_spike_interface_demo.ipynb.
Integrated with other scripts to spikesort from raw data collected using open_ephys
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
import matplotlib.pyplot as plt
import UnitMatchPy.save_utils as su
import UnitMatchPy.GUI as gui
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params


## 

    
def send_test_jobs():
    '''Sends mEC2 first day sessions to be processed.'''
    ephys_df = sps.get_ephys_paths_df()
    ephys_df = ephys_df.sort_values(by=['datetime'])
    for index in [19,20]: #hardcoding here is a bit ugly, but go find a few within day sessions.
        ephys_info = ephys_df.iloc[index]
        if ephys_info['spike_interface_readable'] == True:
            run_e.submit_test_job(ephys_info)
    return print('Submitted a few jobs to test UM outputs')

def test_unit_match():
    ephys_df = sps.get_ephys_paths_df()
    ephys_df = ephys_df.sort_values(by=['datetime'])
    sessions_paths =  []
    for index in [19,20]:
        ephys_info = ephys_df.iloc[index]
        sessions_paths.append(ephys_info['ephys_path'])

def unit_match_sessions(preprocessed_path1, preprocessed_path2):
    '''Code to unit_match sessions, taking inputs in the form of:
    INPUT: list of paths ['./data/preprocessed/subject/session']

    '''

    # default of Spikeinterface as by default spike interface extracts waveforms in a different manner.
    param = {'SpikeWidth': 90, 'waveidx': np.arange(20,50), 'PeakLoc': 35}
    param = default_params.get_default_param()
    all_session_paths = [(Path(preprocessed_path1)/'UM_inputs'),
                         (Path(preprocessed_path2)/'UM_inputs')]
    param['session_paths'] = all_session_paths
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(all_session_paths)
    print('reading Raw waveform data')
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths, unit_label_paths, param, good_units_only = False) 

    #param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

    # create clus_info, contains all unit id/session related info
    clus_info = { 'session_switch' : session_switch, 
                'session_id' : session_id, 
                'original_ids' : np.concatenate(good_units) }

    #Extract parameters from waveform
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    print(extracted_wave_properties)
    #Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors  = ov.extract_metric_scores(extracted_wave_properties, session_switch, within_session, param, niter  = 2)

    #Probability analysis
    prior_match = 1 - (param['n_expected_matches'] / param['n_units']**2 ) # freedom of choose in prior prob
    priors = np.array((prior_match, 1-prior_match))

    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = param['score_vector']
    parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one = 1)

    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

    output_prob_matrix = probability[:,1].reshape(param['n_units'],param['n_units'])

    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

   #save out extracted_wave_properties 
    return extracted_wave_properties

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