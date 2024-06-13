"""Script for trouble shooting KS4"""

# %% Imports

import numpy as np
from kilosort import io, run_kilosort
from spikeinterface import extractors as se
from probeinterface import ProbeGroup, write_prb

# %% Functions


def run_Kilosort4(ephys_path, AP_stream_name, Kilosort_dir):
    """Run kilosort4 without spikeinterface (current bug in SI code, adapt when fixed)"""
    # if sorter_output already exists, load and return
    internal_path = "Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-100.mEC8-AP/continuous.dat"
    raw_data_path = ephys_path / internal_path
    raw_rec = se.read_openephys(ephys_path, stream_name=AP_stream_name)
    sorter_output_dir = Kilosort_dir / "sorter_output"

    n_channels = raw_rec.channel_ids.shape[0]
    sample_freq = raw_rec.sampling_frequency
    settings = {"fs": sample_freq, "n_chan_bin": n_channels}
    probe = _load_Kilosort4_probe(raw_rec, Kilosort_dir)
    outputs = run_kilosort(
        settings=settings,
        probe=probe,
        filename=raw_data_path,
        results_dir=sorter_output_dir,
        data_dtype="int16",
        do_CAR=True,
    )
    return


def _load_Kilosort4_probe(si_recording, temp_output_folder):
    """
    This function saves a spikeinterface probe object to a .prb file that can be loaded
    into Kilosort4

    Inputs:
    - si_recording: spikeinterface recording object
    - temp_output_folder: Path to save the .prb file

    Returns:
    - KS probe dict
    """
    pg = ProbeGroup()
    pg.add_probe(probe=si_recording.get_probe())
    probe_path = temp_output_folder / "probe.prb"
    write_prb(probe_path, pg)
    return io.load_probe(probe_path)
