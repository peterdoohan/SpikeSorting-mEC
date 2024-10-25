# Preprocessing Maze Ephys Data with SpikeInterface & Kilosort4

## STEP 1: Create a new environment
- create new conda environment 
    ``` bash 
    conda create --name maze_ephys python==3.10 
    conda activate maze_ephys
    ```
- install required packages (assuming you're in .../code)
    ``` bash 
    pip install -r SpikeSorting/requirements.txt
    ```
- install spikeinterface
    ``` bash
    pip install spikeinterface==0.101.0
    ```
- also install pandas and matplotlib 

## STEP 2: Organise Your Ephys Data

 - We're expecting data which is structured as follows:
    ```python
    .
    └── Experiment/ 
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
        cd <.../code>
    ```
    
## STEP 3: Perform initial SpikeInterface and kilosort checks

 - Channel assignments:
    For some recordings the probe may not be entirely within the brain.
    We therefore double-check that outside-brain channels are identified. 
    Code to do so is given under notebooks/preprocessing_ephys.ipynb.

 - Optimise kilosort:
    This is a ball-park estimation made by slightly tweaking the Th parameters as suggested.
    (https://kilosort.readthedocs.io/en/stable/parameters.html)
    We use data from the first and last recordings of each subject.

    ```bash
    ipython
    import optimise_kilosort as ok
    ok.submit_jobs()
    #then after finishing
    ok.summarise_results()
    ```
    the summary plots are found under data/preprocessing/spikesorting/kilosort_optim.