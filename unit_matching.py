"""A script to unitmatch for tracking cells across sessions (within and across days).
Primarily based off of UMPy_spike_interface_demo.ipynb.
Integrated with other scripts to spikesort from raw data collected using open_ephys
@charlesdgburns"""

import os
from pathlib import Path
from datetime import date#


from . import spikesort_session as sps
from . import run_ephys_preprocessing as run_e


## 

    
def send_test_jobs():
    '''Sends mEC2 first day sessions to be processed.'''
    ephys_df = sps.get_ephys_paths_df()
    ephys_df = ephys_df.sort_values(by=['datetime'])
    for index in [17,19,20]: #hardcoding here is a bit ugly, but go find a few within day sessions.
        ephys_info = ephys_df.iloc[index]
        run_e.submit_test_job(ephys_info)
    return print('Submitted a few jobs to test UM outputs')
