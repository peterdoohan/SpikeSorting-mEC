"""Here we compare different parameters for kilosort to use across the dataset; the preferred parameters are then stored for future processing.
@charlesdgburns"""

# %% Imports
import os
from pathlib import Path
from datetime import date

from . import spikesort_session as sps
from . import run_ephys_preprocessing as rep

## Global variables
SPIKESORTING_PATH = Path("../data/preprocessed_data/spikesorting") 
# INPUT should only be the range of dates for the experiment. 
# Define the date range
START_DATE = date.fromisoformat('2024-02-20')
END_DATE = date.fromisoformat('2024-04-15')

# We then want to run kilosort with a few test parameters
# on the longest session of each subject on the first and last day of the experiment.


# %% functions

def submit_jobs():
    '''Submits a bunch of jobs via SLURM to kilosort each session at each set of parameters.'''
    sample_paths_df = get_sample_paths_df() #Get first and last session info

    #Make a separate directory for each set of Th parameters.
    for subfolder, kilosort_Ths in zip(['lower','default','higher'],[[7,6],[9,8],[11,10]]):
        spikesorting_path = SPIKESORTING_PATH/'kilosort_optim'/subfolder
        if not spikesorting_path.exists():
            spikesorting_path.mkdir(parents=True)

        print(f"For {subfolder} kilosort parameter settings ({kilosort_Ths}):")
        for each_session in range(len(sample_paths_df)):
            ephys_info = sample_paths_df.iloc[each_session]
            #We want to submit a bunch of jobs
            script_path = rep.get_ephys_preprocessing_SLURM_script(ephys_info, 
                                                              kilosort_Ths = kilosort_Ths,
                                                              spikesort_path=spikesorting_path)
            os.system(f"chmod +x {script_path}")
            os.system(f"sbatch {script_path}")
            print(f"Test job submitted for {ephys_info.subject_ID} {ephys_info.datetime.isoformat()}")

def submit_test():
    subfolder = 'higher'
    kilosort_Ths = [11,10]
    spikesorting_path = SPIKESORTING_PATH/'kilosort_optim'/subfolder
    if not spikesorting_path.exists():
        spikesorting_path.mkdir(parents=True)
  
    print(f"For {subfolder} kilosort parameter settings ({kilosort_Ths}):")
    
    ephys_info = get_sample_paths_df().iloc[0]
    #Submit test job
    script_path = rep.get_ephys_preprocessing_SLURM_script(ephys_info, 
                                                        kilosort_Ths = kilosort_Ths,
                                                        spikesort_path=spikesorting_path)
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")
    print(f"Test job submitted for {ephys_info.subject_ID} {ephys_info.datetime.isoformat()}")

def get_sample_paths_df():
    '''INPUT: the start and end date of the experiment, in datetime format.
        OUTPUT: paths_df for the longest readable session on the first and last day for each mouse.
    '''
    df = sps.get_ephys_paths_df()
    df['date'] = df['datetime'].apply(lambda x: x.date())  # Create a column of dates
    df_filtered = df[((df['date']==START_DATE) | 
                      (df['date']==END_DATE)) & 
                      (df['spike_interface_readable']==True)]
    # Get the row with the maximum duration_min for each of the two dates (start and end)
    df_filtered['date'] = df_filtered['datetime'].dt.date #ignore time
    sample_paths_df = df_filtered.loc[df_filtered.groupby(['subject_ID','date'])['duration_min'].idxmax()]
    return sample_paths_df
