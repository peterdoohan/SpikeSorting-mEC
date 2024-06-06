# Ephys Preprocessing for GridMaze Experiments
This repo contains code and instructions for spike sorting ephys data from gridmaze experiments using [spikeinterface]() and [Kilosort](). To go from raw ephys data stored on the cluster through preprocessing, spikesorting and quality control you'll need to complete the following:
- [Set up your python environment](#create-spike_sorting-environment)
- [Test Pipeline on some example sessions](#test-the-pipeline-on-a-few-sessions)
- [Run ephys preprocessing on an entire dataset](#run-pipeline-on-all-sessions)

## Create ```spike_sorting``` Environment
To avoid package conflicts lets make a new conda environment specific for spike_sorting using the code from this repo (Note Kilsort4 suggests using python version 3.9)

Create environment, activiate and install required packages, with the following terminal commands:
```
conda create --name spike_sorting python=3.9
``` 
```
conda activate spike_sorting
```
```
pip install spikeinterface kilosort==4.0.6
```

### Add local version of Kilosort3
If you want to use Kilosort3 for spike sorting you will need to clone and compile the code locally. Note it is possible to avoid doing this by using spikeinterface's [containerised spike sorting system](https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html#running-sorters-in-docker-singularity-containers), but I haven't got this to work on the cluster yet. 

Lets clone the Kilosort to our project's code folder, swap versions to Kilosort3 (and rename the folder to avoid confusion), and do some prerequist compiling

```
cd <your_project/code>
git clone https://github.com/MouseLand/Kilosort.git
```
```
git checkout v3.0.2
mv Kilosort Kilosort3

```
```
module load matlab/R2021a
module load cuda/11.8
cd Kilosort3/CUDA
matlab
mexGPUall
```
You should now be read to use your local installation of Kilosort3 with Spike Interface. Your code folder should look the same as in the [example directory system](#example-directory-system).



## Test pipeline on a few sessions



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

