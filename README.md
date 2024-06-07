# Ephys Preprocessing for GridMaze Experiments
This repo contains code and instructions for spike sorting ephys data from gridmaze experiments using [spikeinterface]() and [Kilosort](). To go from raw ephys data stored on the cluster through preprocessing, spikesorting and quality control you'll need to complete the following:
- [Set up your python environment](#create-spike_sorting-environment)
- [Test Pipeline on some example sessions](#test-the-pipeline-on-a-few-sessions)
- [Run ephys preprocessing on an entire dataset](#run-pipeline-on-all-sessions)

## Create a ```spike_sorting``` Environment
To avoid package conflicts lets make a new conda environment specific for spike_sorting using the code from this repo (Note Kilsort4 suggests using python version 3.9)

Create environment, activiate and install required packages, with the following terminal commands:
```shell
conda create --name spike_sorting python=3.9
``` 
```shell
conda activate spike_sorting
```
```shell
pip install spikeinterface kilosort==4.0.6
```

### Add local version of Kilosort3
If you want to use Kilosort3 for spike sorting you will need to clone and compile the code locally. Note it is possible to avoid doing this by using spikeinterface's [containerised spike sorting system](https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html#running-sorters-in-docker-singularity-containers), but I haven't got this to work on the cluster yet. 

Lets clone the Kilosort to our project's code folder, swap versions to Kilosort3 (and rename the folder to avoid confusion), and do some prerequist compiling

```shell
cd <your_project/code>
git clone https://github.com/MouseLand/Kilosort.git
```
```shell
mv Kilosort Kilosort3
cd Kilosort3
git checkout v3.0.2
```
```shell
module load matlab/R2021a
module load cuda/11.8
cd CUDA
matlab
mexGPUall
```
You should now be read to use your local installation of Kilosort3 with Spike Interface. Your code folder should look the same as in the [example directory system](#example-directory-system).



## Test pipeline on a few sessions
The python library in this repo, spikesort_session.py, contains functions to run and test the ephys preprocessing pipeline.

### Reading OpenEphys session folders with spikeinterface
- spikeinterface offers a host of tools for loading raw ephys data, we can load our files with:
    ```python
    from spikeinterface import extractors as se
    raw_rec = se.read_openephys(raw_rec)
    ```
    - Note however, that with Neuropixel 1.0, AP and LFP bands are saved separately and will need to speficifed with a stream_name parameter.
- At least in my case, errors occured during ephys data aquisition that means that some sessions were aborted, contain errors, multiple recordings folders when recording had to be restarted etc. To get an overview of all my sessions and which sessions don't contain errors and therefore can be loaded easy with spikeinterface:
    ```python
    import SpikeSorting as sss
    ephys_paths_df = sss.get_ephys_paths_df()
    ```
    - This generates a dataframe that finds the right ephys_path, AP_stream_name, etc. for a session and also checks if the session can be loaded with spikeinterface (see function docstring). 
- From this dataframe (or your own version of it), we can manually select a few ephys_path (maybe a few from each subject) to test the spike sorting pipeline on.
- If you're in the middle of an experiment and just want to check a few sessions you can, of course, manually select some files names from ```raw_data/ephys``` to play with!

### Running the spike sorting pipeline on a single session


## Run pipeline on all sessions



## Example directory system
```
ProjectFolder/
└── experiment/
    ├── code/
    │   ├── SpikeSorting/
    │   │   └── spikesort_session.py
    │   └── Kilosort3
    ├── data/
    │   ├── raw_data
    │   └── preprocessed_data/
    │       └── Kilosort/
    │           ├── subject_1
    │           ├── subject_2/
    │           │   ├── session_datetime_1/
    │           │   │   └── Kilosort3or4/
    │           │   │       ├── sorter_output
    │           │   │       └── report/
    │           │   │           └── units/
    │           │   │               ├── unit0.png
    │           │   │               ├── unit1.png
    │           │   │               └── ...
    │           │   ├── session_datetime_2
    │           │   └── ...
    │           └── ...
    └── results
```

