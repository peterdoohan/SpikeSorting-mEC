"""A script to unitmatch for tracking cells across sessions (within and across days).
Integrated with other scripts to spikesort from raw data cole
@charlesdgburns"""

import os
from pathlib import Path
from datetime import date
import UnitMatchPy
from . import spikesort_session as sps
from . import run_ephys_preprocessing as run_e

## 

def store_test_snippets():
    

def send_test_jobs():
    '''Sends mEC2 first day sessions to be processed.'''
    ephys_df = sps.get_ephys_paths_df()
    ephys_df = ephys_df.sort_values['datetime']#
    for index in [17,19,20]:
        ephys_info = ephys_df.iloc[index]
        run_e.submit_test_job(ephys_info)
    return 
