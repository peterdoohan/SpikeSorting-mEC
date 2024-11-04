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

#to help plot reports:
from spikeinterface import core as si
import spikeinterface.widgets as sw



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


def run_unitmatch_pairwise():
    ''' Top level function to run unit match (for each subject) for each pair of sessions.
    Submits jobs to clusters for each subject.'''
    
    for each_subject in sps.get_ephys_paths_df()['subject_ID'].unique():
        # check jobs folder exits
        for jobs_folder in ["slurm", "out", "err"]:
            if not Path(f"SpikeSorting/jobs/{jobs_folder}").exists():
                os.mkdir(f"SpikeSorting/jobs/{jobs_folder}")

        print(f"Submitting {each_subject} pairs to HPC")
        script_path = get_unitmatch_SLURM_script(subject=each_subject)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("All ephys preprocessing jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def get_unitmatch_SLURM_script(subject, RAM="64GB", time_limit="48:00:00"):
    '''Writes out script to perform pairwise unit matching for all pairs of sessions
    for a given subject.'''
    subject_ID = f"{subject}"
    script = f"""#!/bin/bash
#SBATCH --job-name=ephys_preprocessing_{subject_ID}
#SBATCH --output=SpikeSorting/jobs/out/unit_matching_{subject_ID}.out
#SBATCH --error=SpikeSorting/jobs/err/unit_matching_{subject_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

source $(conda info --base)/etc/profile.d/conda.sh
module load miniconda
module load cuda/11.8
conda deactivate
conda activate maze_ephys

python -c \"
import SpikeSorting.unit_matching as um
print('Starting UnitMatch for {subject_ID}')
pairs_df = um.get_pairs_df(subject='{subject_ID}')
um.get_pairwise_matches(pairs_df)
\"
"""
    script_path = f"SpikeSorting/jobs/slurm/unit_matching_{subject_ID}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
    

def get_pairwise_matches(pairs_df):
    '''INPUT: dataframe with sessions to pairwise run unit match on.
        Input should match output of get_pairs_df().
        '''
    for _, pair in pairs_df.iterrows():
        print(f"Matching sessions {pair['datetimes']}")
        UM_out_path = Path(pair['UM_out_path'])
        UM_out_path.mkdir(parents=True,exist_ok=True)
        for preprocessed_path in pair['preprocessed_paths']:
            sps.save_unitmatch_labels(preprocessed_path)
            pad_UM_inputs(preprocessed_path)
        print(f'Output path: {UM_out_path}')
        try: 
            if pair['completed_UM']:
                print('UnitMatching already completed')
            else:
                unit_match_sessions(pair['UM_input_paths'])
            if pair['completed_reports']:
                print('UnitMatch reports already completed')
            else:
                get_unitmatch_reports(pair)

        except Exception as e:
            print('Error running unitmatch. Saving error log to output path.')
            with open((UM_out_path/'failed_unitmatch.txt'), 'a') as f: 
                f.write(str(e) + '\n')     
            
def unit_match_sessions(UM_input_paths, param = PARAM):
    '''Code to match units across sessions, taking inputs in the form of:
    INPUT: list of session path objects: 'SPIKESORTING_PATH/subject/datetime/'UM_inputs'. 
            parameters for unit match (n_shanks, max_dist, channel_radius et.c.)
    OUTPUT: saves 'standard' unit match outputs to a folder in:
            .../preprocessed_data/UnitMatch/subject/datetime1xdatetime2 (longer if more than two sessions)
    '''
    #Double-check all sessions are from the same subject:
    subject_list = [x.parts[-3] for x in UM_input_paths]
    if not all(s == subject_list[0] for s in subject_list):
        raise ValueError('Failed unitmatching. Not all sessions are from the same subject.')
        
    datetimes = [x.parts[-2] for x in UM_input_paths]
    joined_datetimes = 'x'.join(datetimes)
    UM_out_path = Path('../data/preprocessed_data/UnitMatch')/subject_list[0]/f'{joined_datetimes}'
    UM_out_path.mkdir(parents=True,exist_ok=True)
    param['session_paths'] = UM_input_paths
    
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(UM_input_paths)
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
    total_score, candidate_pairs, scores_to_include, predictors  = ov.extract_metric_scores(wave_dict, session_switch, within_session, param, niter = 2)
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
                                                                            best_drift=False) #NB: set to false to apply only basic drift correction


    thrs_opt = mf.get_threshold(total_score, within_session, euclid_dist, param, is_first_pass = False)
    param['n_expected_matches'] = np.sum( (total_score > thrs_opt).astype(int))
    prior_match = 1 - ( param['n_expected_matches'] / len(include_these_pairs))
    print(prior_match)
    if abs(prior_match)<1: #if there's a weird error after drift correction, we ignore it, otherwise
        thrs_opt = np.quantile(total_score[include_these_pairs_idx.astype(bool)], prior_match)
        candidate_pairs = total_score > thrs_opt
    else:
        print(f'Odd value of Prior Match = {prior_match}, so ignoring drift correction')

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

## Unit match report functions


def get_unitmatch_reports(pair):
    '''INPUTS: unitmatch session pair info, from get_pairs_df(). 
        Must contain:
        pair['UM_out_path'],
        pair['preprocessed_paths'],
        pair['datetimes'],
        
       *must also have probe channel locations under 
        '../data/preprocessed_data/spikesorting/probe_params/location.npy'
        
        OUTPUTS: a report for each match *across sessions*, for sanity checking.
    '''
        # Read relevant data:

    print('reading unitmatch outputs for reports')
    UM_out_path = pair['UM_out_path']
    match_df = pd.read_csv(UM_out_path/'MatchTable.csv')
    clus_info = np.load(UM_out_path/'ClusInfo.pickle', allow_pickle=True)
    wave_data = np.load((UM_out_path/'WaveformInfo.npz'))
    channel_pos = np.load('../data/preprocessed_data/spikesorting/probe_params/location.npy')
    match_df = match_df.query('`UM Probabilities`>0.5 and `RecSes 1`!= `RecSes 2`')

    reports_dir = UM_out_path/'match_reports'
    reports_dir.mkdir(exist_ok=True)
    

    if len(match_df) == 0:
        print('No matches across sessions')
        open(reports_dir/"no_cross_matches.txt",'w').close() 
    else:
        print('Saving out a report for each match')
        for index, match_info in match_df.iterrows():
            
            #read out session and unit id data for avg waveform and centroid trajectories
            unit_a_session = int(match_info['RecSes 1']-1)
            unit_b_session = int(match_info['RecSes 2']-1)
            unit_a_mask = (clus_info['original_ids']==match_info['ID1']).T*(clus_info['session_id']==unit_a_session)
            unit_b_mask = (clus_info['original_ids']==match_info['ID2']).T*(clus_info['session_id']==unit_b_session)
            wave_data_idx_a = np.argwhere(unit_a_mask[0]==True)[0][0]
            wave_data_idx_b = np.argwhere(unit_b_mask[0]==True)[0][0]
            wave_data_idxs = [wave_data_idx_a,wave_data_idx_b]
            avg_wave = np.mean(wave_data['avg_waveform'],axis=2) #(n_timesteps, n_clusters), averaging over split halves or 'cv'
            avg_pos = np.mean(wave_data['avg_waveform_per_tp'],axis=3) #(n_coords,n_clusters,n_timesteps)  
            
            #load analyzers for autocorrelograms and spike distributions
            #this is using data computed and stored via spikesort_session.py
            analyzer_a = si.load_sorting_analyzer(pair['preprocessed_paths'][unit_a_session]/'sorting_analyzer')
            analyzer_b = si.load_sorting_analyzer(pair['preprocessed_paths'][unit_b_session]/'sorting_analyzer')
            analyzers = [analyzer_a,analyzer_b]
            analyzer_idxs = [int(match_info['ID1']),int(match_info['ID2'])]
            
            # PLOTTING:
            # Set up the figure and axes.
            fig = plt.figure(figsize=(10, 4))
            subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1,3]) #large column to left,
            axsLeft = subfigs[0].subplots(1,1)
            axs = subfigs[1].subplots(2,2)

            #Add colour coded title text
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] #blue is 0, orange is 1
            subject = pair['preprocessed_paths'][0].parts[-2] #a bit finicky, but inherited from data structure
            
            match_text = f'{round(match_info['UM Probabilities']*100,2)}% match \n {round(match_info['TotalScore'],3)} total score'
            fig.text(0.1,1.05, match_text, 
                    ha="center", va="bottom", size="large")

            text_a = f'{subject}.{pair['datetimes'][unit_a_session]}.cluster_{analyzer_idxs[0]}'
            fig.text(0.4,1.05, text_a, ha="center", va="bottom", size="large",color=colours[0])

            text_b = f'{subject}.{pair['datetimes'][unit_b_session]}.cluster_{analyzer_idxs[1]}'
            fig.text(0.8,1.05, text_b, ha="center", va="bottom", size="large",color=colours[1])


            for each_unit in range(2):
                axsLeft.set(title='Waveform templates')
                sw.plot_unit_waveforms(analyzers[each_unit], plot_waveforms=True, plot_templates=True,
                            alpha_waveforms = 0.001, alpha_templates = 0.5,
                            unit_ids=[analyzer_idxs[each_unit]], 
                            unit_colors={analyzer_idxs[each_unit]:colours[each_unit]}, 
                            set_title=False, plot_legend=False,
                            backend='matplotlib',same_axis=True, **{'ax':axsLeft})
                
                axs[0,0].set(title='Average waveforms')
                axs[0,0].plot(avg_wave[:,wave_data_idxs[each_unit]])
                
                axs[1,0].set(title = 'Average centroid')
                #first we want to plot on a scaffold of channel locations around the centroid
                max_channel_idx = wave_data['max_site'][wave_data_idxs[each_unit],0]
                axs[1,0].scatter(x=channel_pos[(max_channel_idx-4):(max_channel_idx+4),0],
                            y=channel_pos[(max_channel_idx-4):(max_channel_idx+4),1],
                            marker = 's', color='gray')
                axs[1,0].scatter(x=avg_pos[1,wave_data_idxs[each_unit],:],
                            y=avg_pos[2,wave_data_idxs[each_unit],:],
                            alpha=0.3,
                            color = colours[each_unit])
            
                axs[0,1].set(title='Spike amplitude distributions')
                sw.plot_amplitudes(analyzers[each_unit], plot_histograms=False, plot_legend=False,
                            unit_ids=[analyzer_idxs[each_unit]], 
                            unit_colors={analyzer_idxs[each_unit]:colours[each_unit]}, 
                        backend='matplotlib', **{'ax':axs[0,1]} )
                for artist in axs[0,1].collections:
                    artist.set_alpha(0.3)  # Adjust alpha for all scatter points

                #lineplot autocorrelograms
                axs[1,1].set(title='Normalised AutoCorrelogram')
                corr_data = analyzers[each_unit].get_extension('correlograms').get_data()
                bins = corr_data[1]  # Time bins for correlograms
                corr_values = corr_data[0][analyzer_idxs[each_unit], analyzer_idxs[each_unit], :]  # CCG values for the specific unit pair
                corr_normalised = corr_values/max(corr_values)
                axs[1,1].plot(bins[:-1], corr_normalised, linestyle='-', alpha=0.5)

            fig.savefig(reports_dir/f'{text_a}x{text_b}.png', bbox_inches="tight")


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
        for datetime in subject_df['datetime']:
            session_path = sps.SPIKESORTING_PATH/each_subject/datetime.isoformat()
            um_labels = sps.pd.read_csv(session_path/'UM_inputs'/'cluster_group.tsv', sep='\t')
            if sum(um_labels.KSLabel=='good') == 0:
                open(session_path/'UM_inputs'/"no_good_units.txt",'w').close() 
                excluded_datetimes.append(datetime)
    um_paths_df = um_paths_df[~um_paths_df['datetime'].isin(excluded_datetimes)]
    print(f'Excluded {len(excluded_datetimes)} sessions across all subjects due to no good units')
    um_paths_df['date'] = um_paths_df.loc[:,'datetime'].apply(lambda x: x.date())
    
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

def get_pairs_df(subject:str):
    '''INPUT: subject string such as 'mEC_2'.
    OUTPUT: a dataframe with all pairs of sessions with the following columns:#
     'datetimes': (datetime1,datetime2) isoformat tuple.
         but also individual columns for 'datetime1' and 'datetime2'.
     'preprocessed_paths': (preprocessed_path1,preprocessed_path2) tuple.
     'UM_input_paths': (UM_input_path1,UM_input_path2) tuple
     'UM_out_path': .../preprocessed_data/UnitMatch/subject/datetime1xdatetime2 '''
     
    # First get a dataframe with subject and get all pairwise datetime combinations
    um_paths_df = get_um_paths_df()
    subject_df = um_paths_df[um_paths_df['subject_ID']==subject]
    datetimes = list(combinations(subject_df['datetime'].apply(lambda x: x.isoformat()),2))

    pairs_df = pd.DataFrame({'datetimes': datetimes})
    pairs_df['datetime1'] = pairs_df.loc[:,'datetimes'].apply(lambda x: x[0])
    pairs_df['datetime2'] = pairs_df.loc[:,'datetimes'].apply(lambda x: x[1])
    
    # add some useful paths
    pairs_df['preprocessed_paths'] = pairs_df.loc[:,'datetimes'].apply(lambda x: 
        (sps.SPIKESORTING_PATH/subject/x[0],sps.SPIKESORTING_PATH/subject/x[1]))

    pairs_df['UM_input_paths'] = pairs_df.loc[:,'preprocessed_paths'].apply(lambda x:
        [x[0]/'UM_inputs',x[1]/'UM_inputs'])

    pairs_df['UM_out_path'] = pairs_df.loc[:,'datetimes'].apply(lambda x: 
        sps.SPIKESORTING_PATH.parent/'UnitMatch'/subject/f'{x[0]}x{x[1]}')

    # Check whether we've already run unitmatch on a pair
    pairs_df['completed_UM'] = pairs_df.loc[:,'UM_out_path'].apply(lambda x: 
        True if os.path.exists(x/'MatchTable.csv') else False)
    pairs_df['completed_reports'] = pairs_df.loc[:,'UM_out_path'].apply(lambda x: 
        True if os.path.exists(x/'match_reports') else False)
    return pairs_df


## Development // debugging

