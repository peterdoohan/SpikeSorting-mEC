"""Script to spikesort ephys sessions using Kilosort and Spike interface"""
#%%  imports

#%% Global variables

TEST_SESSIONS = [
    "../data/raw_data/ephys/mEC_6/2024-02-20_12-55-26",
    "../data/raw_data/ephys/mEC_6/2024-02-20_14-22-35",
    "../data/raw_data/ephys/mEC_6/2024-02-24_12-42-47",
    "../data/raw_data/ephys/mEC_6/2024-02-24_14-14-18",
    "../data/raw_data/ephys/mEC_6/2024-02-27_12-13-53",
    "../data/raw_data/ephys/mEC_6/2024-02-27_13-46-14",
    "../data/raw_data/ephys/mEC_6/2024-03-09_14-34-06",
    "../data/raw_data/ephys/mEC_6/2024-03-09_16-07-05",
    "../data/raw_data/ephys/mEC_6/2024-03-17_11-30-10",
    "../data/raw_data/ephys/mEC_6/2024-03-17_12-57-33",
    "../data/raw_data/ephys/mEC_6/2024-03-18_12-23-37",
    "../data/raw_data/ephys/mEC_6/2024-03-18_14-15-04",
]

#%% Functions 