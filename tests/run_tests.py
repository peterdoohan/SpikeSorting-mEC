"""
Script to test functions to make sure changes do not break use-cases across probe setups and types.
'.../tests/' will be treated as '.../experiment/code', so that paths map to data correctly.s
@charlesdgburns
"""

#basic utilities

from datetime import date
from pathlib import Path
from distutils.dir_util import copy_tree
#import scripts to test
from .. import spikesort_session as sps
from .. import optimise_kilosort as ok
from .. import run_ephys_preprocessing as run_e
from .. import unit_matching as um

import os
current_directory = Path(os.getcwd())
if current_directory.parts[-1] == 'code':
    os.chdir('./SpikeSorting/tests')
    print('Setting OS directory to tests subfolder.')
elif current_directory.parts[-1] =='tests':
    print('current OS directory set to tests subfolder.')
else:
    raise print('Check directories before import.')

# Here we test functions on example data as if we were to go through the pipeline

# %% set up directories and test paths
sps.SPIKESORTING_PATH.mkdir(parents=True, exist_ok=True)
ephys_paths_df = sps.get_ephys_paths_df()

JOBS_FOLDER = Path('jobs/')

if not JOBS_FOLDER.exists():
    print('creating Jobs folders')
    for sub_folder in ["slurm", "out", "err"]:
        if not (JOBS_FOLDER/sub_folder).exists():
            (JOBS_FOLDER/sub_folder).mkdir(parents=True,exist_ok=True)

# %% PROBE SETUP AND BAD CHANNEL ASSIGNMENT

print('Saving out probe information per subject')
print('Assigning bad channel parameters across first and last recordings')

for subject_ID in ephys_paths_df['subject_ID'].unique():
    if subject_ID in ['subject_01', 'subject_02', 'subject_03']:
        #for neuropixel probes (automatic), and subject_03 cambridgeneurotech setup.
        sps.save_rec_probe(subject_ID,
                            manufacturer = 'cambridgeneurotech',
                            probe_name = 'ASSY-236-F',
                            wiring_device='cambridgeneurotech_mini-amp-64',
                            probe_suffix = None, #'probe_A', 'probe_B' if multiple probes; do this once for each probe, for each subject
                            manual_ephys_path=None) #should handle things automatically, but included here for debugging.
        if subject_ID == 'subject_01': #neuropixel 1 probe with many outside brain channels.
            sps.save_channel_assignment_params(subject_ID,outside_thresh=-0.15, n_neighbours=37)    
        else: #otherwise use default settings
            sps.save_channel_assignment_params(subject_ID, outside_thresh=-0.75, n_neighbours=11) 
    elif subject_ID == 'subject_04': #multi-probe subject
        for each_probe in ['probe_A','probe_B']:
            sps.save_rec_probe(subject_ID,
                                    manufacturer = 'cambridgeneurotech',
                                    probe_name = 'ASSY-236-F',
                                    wiring_device='cambridgeneurotech_mini-amp-64',
                                    probe_suffix = each_probe, #'probe_A', 'probe_B' if multiple probes; do this once for each probe, for each subject
                                    manual_ephys_path=None) #should handle things automatically, but included here for debugging.
            sps.save_channel_assignment_params(subject_ID, outside_thresh = -5, #stricter threshold - we want no bad channels tbh
                                                n_neighbours = 11, probe_suffix = each_probe)
  
# %% Attempt to run kilosort optimisation
ok.submit_jobs(jobs_folder = JOBS_FOLDER, python_path = '../..')

# %% Utility functions

#we want to grab two recordings from each subject / probe setup. 
first_date = '1970-01-01_00-00-00' #midnight on 1st of january 1970. Thanks, openephys.
last_date = '2012-12-21_13-59-47' #end of the mayan calendar?
# %% Functions

def copy_raw_ephys(new_subject_name, new_datetime, ephys_path):
    '''INPUT: ephys_path = './path/to/ephys/subject/datetime/' 
              new_subject_name = 'subject_0X'
    '''
    new_ephys_path = Path(f'../data/raw_data/ephys/{new_subject_name}')/new_datetime
    new_ephys_path.mkdir(parents=True, exist_ok=True)
    copy_tree(ephys_path, new_ephys_path)
    return print('Copied over raw ephys data')

#hard-coding data paths

def copy_all_test_data():
    #Neuropixel 1.0 data single probe
    copy_raw_ephys('subject_01',first_date,'/ceph/behrens/peter_doohan/goalNav_mEC/experiment/data/raw_data/ephys/mEC_5/2024-02-20_10-38-18')
    copy_raw_ephys('subject_01',last_date,'/ceph/behrens/peter_doohan/goalNav_mEC/experiment/data/raw_data/ephys/mEC_5/2024-02-20_12-04-44')
    #Neuropixel 2.0 data single probe
    copy_raw_ephys('subject_02',first_date,'/ceph/behrens/Francesca/compReplay_mEC/experiment/data/raw_data/ephys/MR2_NM/2024-06-22_12-54-16')
    copy_raw_ephys('subject_02',last_date,'/ceph/behrens/Francesca/compReplay_mEC/experiment/data/raw_data/ephys/MR2_NM/2024-07-01_15-59-29')
    #CambridgeNeurotech data single probe
    copy_raw_ephys('subject_03',first_date,'/ceph/behrens/peter_doohan/goalNav_mFC/experiment/data/raw_data/ephys/m2/2022-06-22_14-05-23')
    copy_raw_ephys('subject_03',last_date,'/ceph/behrens/peter_doohan/goalNav_mFC/experiment/data/raw_data/ephys/m2/2022-07-21_13-13-19')               
    #CambridgeNeurotech data dual probe 
    copy_raw_ephys('subject_04',first_date,'/ceph/behrens/Beatriz/beatriz/7x7_Maze_HC_Rec_FEB2023/03_recordings_data/MR34/2023-02-21_15-11-11')
    copy_raw_ephys('subject_04',last_date,'/ceph/behrens/Beatriz/beatriz/7x7_Maze_HC_Rec_FEB2023/03_recordings_data/MR34/2023-03-09_15-48-55')               
   

#STRONG TODO:
#def chop_raw_ephys()

