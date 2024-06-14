"""Library of functions to plot and compare quality metrics output from spikesorting/ephys preprocessing"""

# %% Imports
import os
import pandas as pd
import seaborn as sns
from datetime import date
from datetime import datetime as dt
from . import preprocess_ephys as pe

# %% Functions


def preprocess_test_sessions(
    test_date=date(2024, 2, 22),
    sorters=["Kilosort3"],
    sorter2sorter_params={
        "Kilosort3": {
            "detect_threshold": 6,
            "projection_threshold": [10, 12],
            "car": False,
            "do_correction": True,
        }
    },
):
    """
    Runs ephys preprocessing for different spike sorters on test sessions from one day of recording across all subjects.
    Jobs are submitted to cluster and once run, the output can be compared using plot_qc_metrics.
    """
    ephys_paths_df = pe.get_ephys_paths_df()
    test_sessions_df = ephys_paths_df[ephys_paths_df.datetime.apply(lambda x: x.date()) == test_date]
    for ephys_info in test_sessions_df[1:].itertuples():
        for sorter in sorters:
            print(f"Submitting {ephys_info.subject} {ephys_info.datetime} to HPC")
            script_path = pe.get_ephys_preprocessing_SLURM_script(ephys_info, sorter, sorter2sorter_params[sorter])
            os.system(f"sbatch {script_path}")
    return print("All ephys preprocessing jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def plot_qc_metrics(qc_metrics_paths, labels):
    """
    Plots histograms of QC metrics of Kilosort spike sorting results. For when you may want to compare metrics
    across subjects, preprocessing parameters, spikesorters, etc.

    Args:
        - qc_metrics (list of Path objects): Paths to the output of spikeinterface 'quality metrics.csv'
            in preprocessing/Kilosrot/subject/session/KilosortVersion/report
        - labels (list of str): List of labels to associate with each qc_metrics_path (for plotting)
    """
    # load metric as pd.Dataframes
    qc_metrics_dfs = []
    for qc_path in qc_metrics_paths:
        qc_metrics_df = pd.read_csv(qc_path)
        qc_metrics_df.rename(columns={"Unnamed: 0": "unit_id"}, inplace=True)
        qc_metrics_df.set_index("unit_id", inplace=True)
        qc_metrics_df["amplitude_median"] = qc_metrics_df["amplitude_median"].abs()
        qc_metrics_dfs.append(qc_metrics_df)
    # plot metrics histograms
    colors = sns.palettes.color_palette("tab10", n_colors=len(labels))
    melted_dfs = []
    for df, label in zip(qc_metrics_dfs, labels):
        melted_df = df.melt(var_name="metric", value_name="value")
        melted_df["label"] = label
        melted_dfs.append(melted_df)
    combined_df = pd.concat(melted_dfs)
    hist_plot = sns.FacetGrid(
        combined_df, col="metric", col_wrap=5, hue="label", palette=colors, sharex=False, sharey=False
    )
    hist_plot.map(sns.histplot, "value", bins=50, edgecolor=None, alpha=0.2, multiple="stack", kde=True)
    hist_plot.add_legend()
    # plot sinlge unit bar plots
    n_clusters_df = pd.DataFrame(columns=["total_clusters", "single_units"], index=labels)
    for df, label in zip(qc_metrics_dfs, labels):
        single_units = pe.get_single_units(df)
        n_clusters_df.loc[label, "total_clusters"] = len(df)
        n_clusters_df.loc[label, "single_units"] = len(single_units)
    n_clusters_df.reset_index(inplace=True)
    df_melted = n_clusters_df.melt(id_vars="index", var_name="clusters", value_name="value")
    df_melted.rename(columns={"index": "sorter"}, inplace=True)
    n_clusters_plot = sns.catplot(
        data=df_melted, kind="bar", x="sorter", y="value", hue="clusters", palette="muted", height=6, aspect=1.5
    )
    # Set the title and labels
    n_clusters_plot.set_axis_labels("Sorter", "Count")
    n_clusters_plot.despine(left=True)
    n_clusters_plot.legend.set_title("clusters")
    return hist_plot, n_clusters_plot
