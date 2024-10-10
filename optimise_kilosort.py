"""Here we compare different parameters for kilosort to use across the dataset; the preferred parameters are then stored for future processing.
@charlesdgburns"""

# %% Imports
import os
from pathlib import Path
from datetime import date

from . import spikesort_session as sps
from . import run_ephys_preprocessing as rep

#for plotting
import pandas as pd
import seaborn as sns

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
    for subfolder, kilosort_Ths in zip(['lower','default','higher', 'highest'],[[7,6],[9,8],[11,10],[13,12]]):
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

#Functions for plotting

    
def output_reports():
    '''Outputs summary plots of the optimisation.
       For both the first and last sessions across mice:
       -n_unit_summary.png plots number of kilosort 'good' units and quality metric IBL 'single' units.
       -qual_metric_dist.png plots kde lineplots for the main quality metrics.
    '''
    optim_df = get_optim_df()
    #First a little check that we're done with all the 
    if not optim_df['completed'].all():
        print('Still not finished running all sessions.')
    else:
        optim_df = get_optim_df()
        plot_unit_counts(optim_df)
        plot_qual_metrics_dist(optim_df, single_units_only=False)
        plot_qual_metrics_dist(optim_df, single_units_only=True)
    
    #First get a dataframe 

def get_optim_df():
    '''Generates dataframe for the optimisation process.
    This is based off of directories, which checks progress and is useful for later.'''
    optim_info = []
    for subfolder in ['lower','default','higher','highest']:
        spikesorting_path = sps.SPIKESORTING_PATH/'kilosort_optim'/subfolder   
        for each_subject in os.listdir(spikesorting_path):
            for each_session in os.listdir((spikesorting_path/each_subject)):
                completed = (spikesorting_path/each_subject/each_session/'DONE.txt').exists()
                optim_info.append({'session_path':spikesorting_path/each_subject/each_session,
                                'ks_params':subfolder,
                                'completed':completed })
    optim_df = sps.pd.DataFrame(optim_info)
    return optim_df

def plot_unit_counts(optim_df):
    '''INPUT: optim_df with info about kilosort parameters
       OUTPUT: a png in ../data/preprocessing/spikesorting/optimise_kilosort.'''
    # Count 'good' and 'single' units and then plot comparisons
    optim_df['subject_ID'] = optim_df['session_path'].apply(lambda x: x.parts[-2])
    optim_df['session'] = optim_df['session_path'].apply(lambda x: 'first' if x.parts[-1][:7]=='2024-02' else 'last')

    n_good = []
    n_single = []
    for each_session in optim_df['session_path']:
        ks_labels = sps.pd.read_csv(each_session/'kilosort4'/'sorter_output'/'cluster_KSLabel.tsv', sep='\t')
        good_clusters = ks_labels[ks_labels.KSLabel=='good']
        n_good.append(len(good_clusters))
        qual_df = sps.pd.read_csv(each_session/'quality_metrics.htsv', sep='\t')
        single_units = sps.get_single_units(qual_df)
        n_single.append(len(single_units))

    optim_df['n_good'] = n_good
    optim_df['n_single'] = n_single
    plot_df = optim_df.melt(id_vars=['session_path','ks_params','subject_ID','session'],
                value_vars=['n_good','n_single'], var_name='inclusion', value_name='count')

    import seaborn as sns
    g = sns.FacetGrid(plot_df, col='session', row='inclusion', hue='subject_ID')
    g.map(sns.scatterplot, "ks_params", "count")
    g.add_legend()
    g.savefig(SPIKESORTING_PATH/'kilosort_optim'/'unit_counts.png')
    
    return print('Saved summary plot')

def plot_qual_metrics_dist(optim_df, 
                           single_units_only=False, 
                           per_subject = False):
    '''Plots distribution of quality metrics across clusters.
    -Options to plot after passing exclusion criteria (single units only)
     or to make a panel per subject rather than average across them.'''
    #We make a big pandas dataframe, melt it, then plot it results

    sns.set_context('paper')
    big_metrics_df = pd.DataFrame()
    for each_subject in optim_df['subject_ID'].unique():
        subject_df = optim_df[optim_df['subject_ID']==each_subject]
        for each_session in subject_df.iloc:
            cluster_metrics_df = sps.pd.read_csv((each_session['session_path']/'quality_metrics.htsv'), sep='\t')
            cluster_metrics_df['subject_ID'] = each_session['subject_ID']
            cluster_metrics_df['session'] = each_session['session']
            cluster_metrics_df['ks_params'] = each_session['ks_params']
            big_metrics_df = pd.concat([big_metrics_df, cluster_metrics_df])

    #filter out metric names then melt column
    identifiers = ['subject_ID','session', 'unit_id','ks_params']
    metric_names = ['amplitude_median','firing_rate','presence_ratio', 'isi_violations_ratio']
    if single_units_only == True:
        #filter big_metrics_df
        filter_query = f"isi_violations_ratio <0.1 and amplitude_cutoff<0.1 and firing_rate > 0.1 and presence_ratio>0.9 and amplitude_median>50"
        big_metrics_df = big_metrics_df.query(filter_query)
    big_melt = big_metrics_df.melt(id_vars=identifiers, value_vars=metric_names,
                                var_name='metric')

    #seaborn does some magic from here on:
    if per_subject == True:
        for each_subject in optim_df['subject_ID'].unique():#
            print(f'Plotting for subject {each_subject}')
            subject_df = big_melt[big_melt['subject_ID']==each_subject]   
            g = sns.FacetGrid(subject_df, col='metric',row='session', hue='ks_params', palette='rocket_r',
                                sharey=False, 
                                sharex=False)
            g.map(sns.kdeplot, 'value')
            g.add_legend()
            if single_units_only==True:
                filename = SPIKESORTING_PATH/'kilosort_optim'/f'distributions_single_units_{each_subject}.png'
            else:
                filename = SPIKESORTING_PATH/'kilosort_optim'/f'distributions_all_clusters_{each_subject}.png'
            g.savefig(filename)
    else:
        g = sns.FacetGrid(big_melt, col='metric',row='session', hue='ks_params', palette='rocket_r',
                            sharey=False, 
                            sharex=False)
        g.map(sns.kdeplot, 'value')
        g.add_legend()#
        if single_units_only==True:
            filename = SPIKESORTING_PATH/'kilosort_optim'/'distributions_single_units_all_subs.png'
        else:
            filename = SPIKESORTING_PATH/'kilosort_optim'/'distributions_all_clusters_all_subs.png'
        g.savefig(filename)
    return print('Saved quality metric distribution plot')

## testing / dev functions
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
