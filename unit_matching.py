"""A script to unitmatch for tracking cells across sessions (within and across days).
Primarily based off of UMPy_spike_interface_demo.ipynb.
Integrated with other scripts to spikesort from raw data collected using open_ephys.
Notably, inherits paths from spikesort_session.
@charlesdgburns"""

import os
from pathlib import Path
from datetime import date
from itertools import combinations


from . import spikesort_session as sps
from . import run_ephys_preprocessing as run_e

#import UnitMatchPy
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.metric_functions as mf
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
PARAM['shank_dist'] = 175 #Neuropixel 1.0 probe // SHOULD BE THE DISTANCE WITHIN WHICH YOU CONSIDER A CENTROID TO BE WITHIN A CHANNEL
#Changing distances and radii might be useful for across-day matching.
PARAM['max_dist'] = 250 #default for Neuropixel 2.0 is 100
PARAM['channel_radius'] = 300 #default for Neuropixel 2.0 is 150
PARAM['max_n_channels'] = 384 #Set to Neuropixel 1.0; this is used for padding when channels are removed before spikesorting.

## TODO / NOTES
# There's currently a few sessions which don't match well. 
# Matching across months the drift correction leads to prior_match values around -10, which messes up the quantile for candidate_pairs threshold


def run_pairwise_unit_match(um_paths_df):
    '''INPUT: dataframe with sessions to pairwise run unit match on.
        Top level function to submit a bunch of jobs for unit matching.'''

    #Pair-wise iterating script
    for each_subject in um_paths_df['subject_ID'].unique():
        subject_df = um_paths_df[um_paths_df['subject_ID']==each_subject]
        sessions = subject_df['datetime'].apply(lambda x: x.isoformat())
        session_pairs = list(combinations(sessions,2))
        for pair in session_pairs:
            print(f"Session pair: {pair[0]} and {pair[1]}")
            UM_out_path = Path('../data/preprocessed_data/UnitMatch')/each_subject/f'{pair[0]}x{pair[1]}'
            UM_out_path.mkdir(parents=True,exist_ok=True)
            preprocessed_path1= sps.SPIKESORTING_PATH/each_subject/pair[0]
            preprocessed_path2= sps.SPIKESORTING_PATH/each_subject/pair[1]
            for preprocessed_path in [preprocessed_path1,preprocessed_path2]:
                sps.save_unitmatch_labels(preprocessed_path)
                pad_UM_inputs(preprocessed_path)
            print(f'Output path: {UM_out_path}')
            try: 
                unit_match_sessions(each_subject, pair[0], pair[1])
            except Exception as e:
                print('Error running unitmatch. Saving error log to output path.')
                with open((UM_out_path/'failed_unitmatch.txt'), 'a') as f: 
                    f.write(str(e) + '\n') 
                
            #um.get_match_reports(each_subject, pair[0],pair[1])
            
def unit_match_sessions(paths_UM_inputs, param = PARAM):
    '''Code to match units across sessions, taking inputs in the form of:
    INPUT: list of session path objects: 'SPIKESORTING_PATH/subject/datetime/'UM_inputs'. 
            parameters for unit match (n_shanks, max_dist, channel_radius et.c.)
    OUTPUT: saves 'standard' unit match outputs to a folder in:
            .../preprocessed_data/UnitMatch/subject/datetime1xdatetime2 (longer if more than two sessions)

    '''
    #Double-check all sessions are from the same subject:
    subject_list = [x.parts[-3] for x in paths_UM_inputs]
    if not all(s == subject_list[0] for s in subject_list):
        raise print('Failed unitmatching. Not all sessions are from the same subject.')
        
    datetimes = [x.parts[-2] for x in paths_UM_inputs]
    joined_datetimes = 'x'.join(datetimes)
    UM_out_path = Path('../data/preprocessed_data/UnitMatch')/subject_list[0]/f'{joined_datetimes}'
    UM_out_path.mkdir(parents=True,exist_ok=True)
    param['session_paths'] = paths_UM_inputs
    
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(paths_UM_inputs)
    print('reading Raw waveform data')
    
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths, 
                                                                                                       unit_label_paths, 
                                                                                                       param, 
                                                                                                       good_units_only = True) #this can break if = False; unit match doesn't detect bad units itself.

    #param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

    # create clus_info, contains all unit id/session related info
    clus_info = { 'session_switch' : session_switch, 
                'session_id' : session_id, 
                'original_ids' : np.concatenate(good_units) }

    #Extract parameters from waveform into a wave properties dictionary
    wave_dict = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    #Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors  = extract_metric_scores(wave_dict, session_switch, within_session, param, niter = 2)
    #Probability analysis
    output_prob_matrix = get_output_prob_matrix(param,total_score, candidate_pairs, scores_to_include, predictors)
    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)
    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > 0.75] = 1 #might want to set match threshold to something other than 0.75.
    matches = np.argwhere( ((output_threshold * within_session)) == True) #exclude within session matches
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)
    su.save_to_output(UM_out_path, scores_to_include, matches, output_prob_matrix, 
                    wave_dict['avg_centroid'], 
                    wave_dict['avg_waveform'], 
                    wave_dict['avg_waveform_per_tp'], 
                    wave_dict['max_site'],
                   total_score, output_threshold, clus_info, 
                   param, UIDs = UIDs, matches_curated = None, save_match_table = True) #options
    #save out extracted_wave_properties 
    return print(f'Completed Unitmatch. Saved outputs to: {UM_out_path}')

def get_unitmatch_reports(subject = str, datetime1 = str, datetime2 = str):
    UM_out_path = Path('../data/preprocessed_data/UnitMatch')/subject/f'{datetime1}x{datetime2}'
    

## Unit match subfunctions 

def extract_metric_scores(extracted_wave_properties, session_switch, within_session, param, niter  = 2):
    """
    This function runs all of the metric calculations and drift correction to calculate the probability
    distribution needed for UnitMatch.

    Parameters
    ----------
    extracted_wave_properties : dict
        The extracted properties from extract_parameters()
    session_switch : ndarray
        An array which indicates when anew recording session starts
    within_session : ndarray
        The array which gives each unit a label depending on their session
    param : dict
        The param dictionary
    niter : int, optional
        The number of pass through the function, 1 mean no drift correction
            2 is one pass of drift correction, by default 2

    Returns
    -------
    ndarrays
        The total scores and candidate pairs needed for probability analysis
    """

    #unpack need arrays from the ExtractedWaveProperties dictionary
    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    spatial_decay_fit = extracted_wave_properties['spatial_decay_fit']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    avg_centroid = extracted_wave_properties['avg_centroid']

    #These scores are NOT effected by the drift correction
    amp_score = mf.get_simple_metric(amplitude)
    spatial_decay_score = mf.get_simple_metric(spatial_decay)
    spatial_decay_fit_score = mf.get_simple_metric(spatial_decay_fit, outlier = True)
    wave_corr_score = mf.get_wave_corr(avg_waveform, param)
    wave_mse_score = mf.get_waveforms_mse(avg_waveform, param)

    #affected by drift
    for i in range(niter):
        avg_waveform_per_tp_flip = mf.flip_dim(avg_waveform_per_tp, param)
        euclid_dist = mf.get_Euclidean_dist(avg_waveform_per_tp_flip, param)

        centroid_dist, centroid_var = mf.centroid_metrics(euclid_dist, param)

        euclid_dist_rc = mf.get_recentered_euclidean_dist(avg_waveform_per_tp_flip, avg_centroid, param)

        centroid_dist_recentered = mf.recentered_metrics(euclid_dist_rc)
        traj_angle_score, traj_dist_score = mf.dist_angle(avg_waveform_per_tp_flip, param)


        # Average Euc Dist
        euclid_dist = np.nanmin(euclid_dist[:,param['peak_loc'] - param['waveidx'] == 0, :,:].squeeze(), axis = 1 )

        # TotalScore
        include_these_pairs = np.argwhere( euclid_dist < param['max_dist']) #array indices of pairs to include
        include_these_pairs_idx = np.zeros_like(euclid_dist)
        include_these_pairs_idx[euclid_dist < param['max_dist']] = 1 

        # Make a dictionary of score to include
        centroid_overlord_score = (centroid_dist_recentered + centroid_var) / 2
        waveform_score = (wave_corr_score + wave_mse_score) / 2
        trajectory_score = (traj_angle_score + traj_dist_score) / 2

        scores_to_include = {'amp_score' : amp_score, 'spatial_decay_score' : spatial_decay_score, 'centroid_overlord_score' : centroid_overlord_score,
                        'centroid_dist' : centroid_dist, 'waveform_score' : waveform_score, 'trajectory_score': trajectory_score }

        total_score, predictors = mf.get_total_score(scores_to_include, param)

        #Initial thresholding
        if (i < niter - 1):
            #get the thershold for a match
            thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass = True)

            param['n_expected_matches'] = np.sum( (total_score > thrs_opt).astype(int))
            prior_match = 1 - ( param['n_expected_matches'] / len(include_these_pairs))
            candidate_pairs = total_score > thrs_opt
            drifts, avg_centroid, avg_waveform_per_tp = mf.drift_n_sessions(candidate_pairs, 
                                                                            session_switch, 
                                                                            avg_centroid, 
                                                                            avg_waveform_per_tp, 
                                                                            total_score, 
                                                                            param, 
                                                                            best_drift=True) #NB: set to false to apply only basic drift correction


    thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass = False)
    param['n_expected_matches'] = np.sum( (total_score > thrs_opt).astype(int))
    prior_match = 1 - ( param['n_expected_matches'] / len(include_these_pairs))
    if abs(prior_match)<1: #if there's a weird error after drift correction, we ignore it, otherwise
        thrs_opt = np.quantile(total_score[include_these_pairs_idx.astype(bool)], prior_match)
        candidate_pairs = total_score > thrs_opt

    return total_score, candidate_pairs, scores_to_include, predictors     

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

## Local unit match utilities 

def get_um_paths_df():
    '''Returns a dataframe of sessions valid for unit matching.
    We include completed spike_sorting and non-zero number of 'single' units.'''
    #include only completed sessions
    um_paths_df = sps.get_ephys_paths_df().query('spike_sorting_completed==True')
    #exclude sessions where there are no units for unit matching.
    excluded_datetimes = []
    for each_subject in um_paths_df['subject_ID'].unique():
        subject_df = um_paths_df[um_paths_df['subject_ID']==each_subject]
        subject_df.loc['datetime'] = subject_df['datetime'].apply(lambda x: x.isoformat())
        datetimes = subject_df['datetime']
        for datetime in datetimes:
            print(datetime)
            session_path = sps.SPIKESORTING_PATH/each_subject/str(datetime)
            um_labels = sps.pd.read_csv(session_path/'UM_inputs'/'cluster_group.tsv', sep='\t')
            if sum(um_labels.KSLabel=='good') == 0:
                open(session_path/'UM_inputs'/"no_good_units.txt",'w').close() 
                excluded_datetimes.append(datetime)
    um_paths_df = um_paths_df[~um_paths_df['datetime'].isin(excluded_datetimes)]
    print(f'Excluded {len(excluded_datetimes)} sessions due to no good units')
    um_paths_df.loc['preprocessed_path']=um_paths_df['ephys_path'].apply(lambda x: sps.SPIKESORTING_PATH/Path(x).parts[-2]/Path(x).parts[-1])
    um_paths_df.loc['UM_inputs_path']=um_paths_df['preprocseed_path'].apply(lambda x: x/'UM_inputs')
    um_paths_df.loc['date'] = um_paths_df['datetime'].apply(lambda x: x.date())
    
    return um_paths_df

def pad_UM_inputs(preprocessed_path):
    ''' Function to account for different numbers of channels between sessions,
    due to outside brain channels being removed in preprocessing.'''
    max_n_channels = PARAM['max_n_channels'] #set to neuropixel 1.0 count

    # First we pad positions, shaped [n_channels,n_coords]:
    positions = np.load(preprocessed_path / 'UM_inputs' / 'channel_positions.npy') #(n_channels,2) is the shape here
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