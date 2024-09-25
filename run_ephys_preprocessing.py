"""
Library for submitting ephys preprocessing jobs to the HPC. For main spike-sorting library see spikesort_session.py
@peterdoohan
"""

# %% Imports
import os
from pathlib import Path
from . import spikesort_session as sps


# %% Functions
# %% ephys preprocessing
def run_ephys_preprocessing():
    ephys_paths_df = sps.get_ephys_paths_df()
    assert not ephys_paths_df.empty, "No valid ephys sessions found. Check data paths & get_ephys_paths_df() function."
    # find unspike-sorted sessions
    unsorted_ephys_paths_df = ephys_paths_df[~ephys_paths_df.spike_sorting_completed]
    if unsorted_ephys_paths_df.empty:
        return print("Ephys preprocessing already complete or data not found.")
    # check jobs folder exits
    for jobs_folder in ["slurm", "out", "err"]:
        if not Path(f"SpikeSorting/jobs/{jobs_folder}").exists():
            os.mkdir(f"SpikeSorting/jobs/{jobs_folder}")
    for ephys_info in unsorted_ephys_paths_df.itertuples():
        print(f"Submitting {ephys_info.subject_ID} {ephys_info.datetime} to HPC")
        script_path = get_ephys_preprocessing_SLURM_script(ephys_info)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    return print("All ephys preprocessing jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def submit_test_job(ephys_info):
    script_path = get_ephys_preprocessing_SLURM_script(ephys_info)
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")
    return print(f"Test job submitted for {ephys_info.subject_ID} {ephys_info.datetime}")


def get_ephys_preprocessing_SLURM_script(ephys_info, spike_sorter="Kilosort4", RAM="64GB", time_limit="48:00:00"):
    session_ID = f"{ephys_info.subject_ID}_{ephys_info.datetime}_{spike_sorter}"
    script = f"""#!/bin/bash
#SBATCH --job-name=ephys_preprocessing_{session_ID}
#SBATCH --output=SpikeSorting/jobs/out/ephys_preprocessing_{session_ID}.out
#SBATCH --error=SpikeSorting/jobs/err/ephys_preprocessing_{session_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

module load miniconda
module load cuda/11.8
conda deactivate
conda deactivate
conda deactivate
conda activate maze_ephys

python -c \"
from SpikeSorting.spikesort_session import preprocess_ephys_session
preprocess_ephys_session('{ephys_info.subject_ID}', '{ephys_info.datetime}', '{ephys_info.ephys_path}')
\"
"""
    script_path = f"SpikeSorting/jobs/slurm/ephys_preprocessing_{session_ID}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path
