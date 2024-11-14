# Preprocessing Maze Ephys Data with SpikeInterface & Kilosort4

## STEP 1: Create a new environment
- create new conda environment 
    ``` bash 
    conda create --name maze_ephys python==3.12.7
    conda activate maze_ephys
    ```
- install required packages (assuming you're in .../experiment/code)
    ``` bash 
    pip install -r SpikeSorting/requirements.txt
    ```

## STEP 2: Organise Your Ephys Data

 - We're expecting data which is structured as follows:
    ```python
    .
    └── experiment/ 
        ├── code/ 
        │   └── ... <-- (you are somewhere here)
        └── data/ 
            ├── ... 
            ├── raw_data/
            │   ├── ...
            │   └── ephys/
            │       └── [subject_ID]/
            │           └── [datetime]/
            │               └── [Open Ephys data]...
            └── preprocessed_data/
                ├── ...
                └── spikesorting
                    └── [subject_ID]/
                        └── ...
    ```

 - Make sure your current directory is the 'code' folder
    ``` bash
        cd <.../experiment/code>
    ```
    
## STEP 3: Perform initial SpikeInterface and kilosort checks
 
 - At this stage we recommend going through notebooks/preprocessing_ephys.ipynb.
    To do this on a SLURM server we recommend using a GPU node:
    ```bash
    srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p gpu --gres=gpu:1 --time=12:00:00 --mem=64G --pty bash -i
        #wait for resources to be allocated
    source /etc/profile.d/modules.sh
    module load miniconda
    conda activate maze_ephys     #sometimes requires  'source activate maze_ephys'
    jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888
    ```
    this should print a server URL which can be copied and pasted as an 'existing kernel' in VScode.

 - Recording probe information:
    It is important to manually double-check probe maps for each subject.
    Some recordings might also erroneously be missing this information.
    We therefore store probe information and load it when missing.

    ```bash
    ipython
    ```
    ```python
    import SpikeSorting.spikesort_session as sps
    sps.save_rec_probe(subject_ID, ..., probe_suffix)
    ```
    NB: probe_suffix is only used if there are multiple probes, we recommend 'probe_A', 'probe_B'. This will later be used to count the number of probes per subject.

 - Channel assignments:
    For some recordings the probe may not be entirely within the brain.
    We therefore double-check that outside-brain channels are identified as part of IBL preprocessing. 
    This is semi-automatic, after choosing a set of parameters which work on your subject probes.
 
    ```python
    sps.save_channel_assignment_params(subject_ID, outside_thresh, n_neighbours, probe_suffix)
    ```
    Again, guiding code is given under notebooks/preprocessing_ephys.ipynb.
   
 - Optimise kilosort:
    This is a ball-park estimation made by slightly tweaking the Th parameters as suggested.
    (https://kilosort.readthedocs.io/en/stable/parameters.html)
    We use data from the first and last recordings of each subject.
    - make sure to set these dates in optimise_kilosort.py
    ```bash
    ipython
    import SpikeSorting.optimise_kilosort as ok
    ok.submit_jobs()
    #then after finishing
    ok.summarise_results()
    ```
    the summary plots are found under data/preprocessing/spikesorting/kilosort_optim.
    These plots count the number of 'good' kilosort units and 'single units' passing quality metrics.
    We also plot distributions of quality metrics before/after selection to make sure these are not being biased by Th parameters.
    
    The Th parameters with most single units are saved as best_params.json and used for future kilosorting.
    We also recommend having a careful look at some cluster reports either through the notebook or manually.
    (.../data/preprocessed/optim_kilosort/<subject>/<datetime>/cluster_reports)

    ## STEP 4: Preprocess the whole lot!
    -Submit a lot of jobs to the cluster
       Option: if you want to skip IBL preprocessing based on optimise_kilosort results, we recommend changing the default parameter in get_ephys_preprocessing_SLURM_script() function under run_ephys_preprocessing.py

    ```python
    import SpikeSorting.run_ephys_preprocessing as run_e
    run_e.run_ephys_preprocessing()
    ```
    