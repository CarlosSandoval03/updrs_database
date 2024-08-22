import os
import sys
import copy
from pathlib import Path

import json
import scipy.io
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
from numpy import unique
from numpy import where
from scipy.spatial import distance
from sklearn import cluster
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot


def spectral_dbscan_clustering(data, results_folder, mnum, clustering_type="spectral", n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"], drdr_reduced=False):
    dopamine_resistant = [30, 8, 11, 28, 27, 42, 50, 72, 75, 74, 73, 78, 81, 83]
    dopamine_responsive = [2, 18, 60, 59, 38, 49, 40, 19, 29, 36, 33, 71, 21, 70, 64, 56, 48, 43, 76, 77]
    reduced_subjects = dopamine_resistant + dopamine_responsive
    reduced_subjects = [i - 1 for i in reduced_subjects]

    model = {
        "spectral": SpectralClustering(n_clusters=n_clusters),
        "dbscan":  DBSCAN(eps=0.30, min_samples=9)
    }.get(clustering_type)
    # model = SpectralClustering(n_clusters=n_clusters)
    yhat = model.fit_predict(data)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        if not drdr_reduced:
            plotx = [data[plot_keys[0]][i] for i in row_ix]
            ploty = [data[plot_keys[1]][i] for i in row_ix]
            reduced_str = ""
        else:
            plotx = [data[plot_keys[0]][i] for i in row_ix[0] if i in reduced_subjects]
            ploty = [data[plot_keys[1]][i] for i in row_ix[0] if i in reduced_subjects]
            reduced_str = "34participants"
        pyplot.scatter(plotx, ploty)
    figure_name = os.path.join(results_folder, f"clustering_{clustering_type}_{mnum}_clusters={n_clusters}_{reduced_str}.png")
    plt.xlabel(plot_keys[0])
    plt.ylabel(plot_keys[1])
    plt.title(f"Clustering with {clustering_type}.")
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return yhat


def general_model_clustering(data, results_folder, mnum, clustering_type="gaussian", n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"], drdr_reduced=False):
    dopamine_resistant = [30, 8, 11, 28, 27, 42, 50, 72, 75, 74, 73, 78, 81, 83]
    dopamine_responsive = [2, 18, 60, 59, 38, 49, 40, 19, 29, 36, 33, 71, 21, 70, 64, 56, 48, 43, 76, 77]
    reduced_subjects = dopamine_resistant + dopamine_responsive
    reduced_subjects = [i-1 for i in reduced_subjects]

    model = {
        "gaussian": GaussianMixture(n_components=n_clusters),
        "kmeans": KMeans(n_clusters=n_clusters)
    }.get(clustering_type)

    model.fit(data)
    yhat = model.predict(data)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        if not drdr_reduced:
            plotx = [data[plot_keys[0]][i] for i in row_ix]
            ploty = [data[plot_keys[1]][i] for i in row_ix]
            reduced_str = ""
        else:
            plotx = [data[plot_keys[0]][i] for i in row_ix[0] if i in reduced_subjects]
            ploty = [data[plot_keys[1]][i] for i in row_ix[0] if i in reduced_subjects]
            reduced_str = "34participants"
        pyplot.scatter(plotx, ploty)
    figure_name = os.path.join(results_folder, f"clustering_{clustering_type}_{mnum}_clusters={n_clusters}_{reduced_str}.png")
    plt.xlabel(plot_keys[0])
    plt.ylabel(plot_keys[1])
    plt.title(f"Clustering with {clustering_type}.")
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return yhat


def plot_drdr_scatter_plot_classified(data, results_folder, n_clusters=2, plot_keys=["LogPower", "LimbsRestTrem"]):
    dopamine_resistant = [30, 8, 11, 28, 27, 42, 50, 72, 75, 74, 73, 78, 81, 83]
    dopamine_responsive = [2, 18, 60, 59, 38, 49, 40, 19, 29, 36, 33, 71, 21, 70, 64, 56, 48, 43, 76, 77]

    for sub in dopamine_resistant:
        plotx = data[plot_keys[0]][sub-1]
        ploty = data[plot_keys[1]][sub-1]
        pyplot.scatter(plotx, ploty, color='red', label='Resistant' if sub == dopamine_resistant[0] else "")

    for sub in dopamine_responsive:
        plotx = data[plot_keys[0]][sub-1]
        ploty = data[plot_keys[1]][sub-1]
        pyplot.scatter(plotx, ploty, color='blue', label='Responsive' if sub == dopamine_responsive[0] else "")

    figure_name = os.path.join(results_folder, f"pre-clustering_DRDR_{mnum}_clusters={n_clusters}.png")
    plt.xlabel(plot_keys[0])
    plt.ylabel(plot_keys[1])
    plt.title(f"Predefined DRDR Clustering.")
    plt.legend()
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return 0


def compute_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters
    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn
    X     :  multidimension np array of data points
    Returns:
    -----------------------------------------
    BIC value
    """
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X.iloc[labels == i], [centers[0][i]], 'euclidean')) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return (BIC)


def compute_bic_gpt(kmeans, X):
    k = kmeans.n_clusters
    n = len(X)
    m = X.shape[1]

    # Compute RSS (Residual Sum of Squares)
    rss = np.sum(np.min(kmeans.transform(X) ** 2, axis=1))
    # BIC formula
    bic = n * np.log(rss / n) + k * np.log(n) * m
    return bic


def bic_analysis_clustering(database, max_num_clusters, results_folder):
    num_clusters = range(1, max_num_clusters+1)
    KMeans = [cluster.KMeans(n_clusters=i, init="k-means++").fit(database) for i in num_clusters]

    # now run for each cluster the BIC computation
    BIC_val = [compute_bic_gpt(kmeansi, database) for kmeansi in KMeans]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, max_num_clusters+1),
        y=BIC_val,
        mode='lines+markers',
        marker=dict(size=8, color='blue'),
        line=dict(color='blue'),
        name='BIC'
    ))
    fig.update_layout(
        title='BIC by Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='BIC',
        # template='plotly_white',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(showgrid=True)
    )
    figure_name = os.path.join(results_folder, "BIC", f"bic_by_num-clusters_k-means_clustering.png")
    fig.write_image(figure_name)
    return BIC_val


if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")

    base_drdr_folder = "/project/3024023.01/fMRI_DRDR/"
    results_folder_drdr = "/project/3024023.01/fMRI_DRDR/updrs_analysis/"
    path_to_drdr_data = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean_complete.xlsx")

    if run_local: print("----- LOADING DATABASE PPP -----")
    excel_file_ppp = pd.ExcelFile(path_to_ppp_data_reduced)
    sheets_ppp = excel_file_ppp.sheet_names
    ppp_database = {}
    for sheet in sheets_ppp:
        ppp_database[sheet] = pd.read_excel(path_to_ppp_data_reduced, sheet_name=sheet)

    if run_local: print("----- LOADING DATABASE DRDR -----")
    excel_file_drdr = pd.ExcelFile(path_to_drdr_data)
    sheets_drdr = excel_file_drdr.sheet_names
    drdr_database = {}
    for sheet in sheets_drdr:
        if sheet in ["UPDRS OFF", "UPDRS ON"]:
            drdr_database[sheet] = pd.read_excel(path_to_drdr_data, sheet_name=sheet)
    sheets_drdr = ["UPDRS OFF", "UPDRS ON"]

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    spectral_clustering_flag = False
    k_means_clustering_flag = True
    gaussian_clustering_flag = False
    dbscan_clustering_flag = False
    plot_ground_truth_drdr_flag = False
    bic_flag = False

    # keys_ppp = ["FreqPeak", "LogPower", "TremorUPDRS", "Brady14Items", "LimbsRestTrem", "LimbsRigidity5Items"]
    keys_ppp = ["ResponseTotalU3", "ResponseTremorUPDRS", "ResponseLimbsRestTrem", "ResponseBrady14Items", "ResponseLimbsRigidity5Items"]
    # keys_ppp = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]
    # keys_drdr = ["OFFTremorUPDRS", "AvgBrady14Items", "AvgLimbsRestTrem", "AvgLimbsRigidity5Items"]
    keys_drdr = ["ResponseTotalU3", "ResponseTremorUPDRS", "ResponseLimbsRestTrem", "ResponseBrady14Items", "ResponseLimbsRigidity5Items"]
    # keys_drdr = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]

    data_ppp = ppp_database[sheets_ppp[0]][keys_ppp]
    data_drdr = drdr_database[sheets_drdr[0]][keys_drdr]

    run_ppp = True
    run_drdr = False

    plots_folder_ppp = os.path.join(results_folder_ppp, "clustering")
    plots_folder_drdr = os.path.join(results_folder_drdr, "clustering")
    Path(plots_folder_drdr).mkdir(parents=True, exist_ok=True)
    Path(plots_folder_ppp).mkdir(parents=True, exist_ok=True)

    mnum = "Model_tremor_restremor_only"
    n_clusters = 2
    drdr_reduced = True # Whether I consider only 34 sub or all of them
    # plot_keys_ppp = ["LogPower", "LimbsRestTrem"]
    plot_keys_ppp = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]
    # plot_keys_drdr = ["OFFTremorUPDRS", "AvgLimbsRestTrem"]
    plot_keys_drdr = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if spectral_clustering_flag:
        if run_local: print("----- SPECTRAL CLUSTERING -----")
        if run_ppp: predicted_s_ppp = spectral_dbscan_clustering(data_ppp, plots_folder_ppp, mnum, "spectral",  n_clusters=n_clusters, plot_keys=plot_keys_ppp, drdr_reduced=False)
        if run_drdr: predicted_s_drdr = spectral_dbscan_clustering(data_drdr, plots_folder_drdr, mnum, "spectral", n_clusters=n_clusters, plot_keys=plot_keys_drdr, drdr_reduced=drdr_reduced)

    if gaussian_clustering_flag:
        if run_local: print("----- GAUSSIAN MIXTURE MODEL CLUSTERING -----")
        if run_ppp: predicted = general_model_clustering(data_ppp, plots_folder_ppp, mnum, "gaussian", n_clusters=n_clusters, plot_keys=plot_keys_ppp, drdr_reduced=False)
        if run_drdr: predicted = general_model_clustering(data_drdr, plots_folder_drdr, mnum, "gaussian", n_clusters=n_clusters, plot_keys=plot_keys_drdr, drdr_reduced=drdr_reduced)

    if k_means_clustering_flag:
        if run_local: print("----- KMEANS CLUSTERING -----")
        if run_ppp: predicted_k_ppp = general_model_clustering(data_ppp, plots_folder_ppp, mnum, "kmeans", n_clusters=n_clusters, plot_keys=plot_keys_ppp, drdr_reduced=False)
        if run_drdr: predicted_k_drdr = general_model_clustering(data_drdr, plots_folder_drdr, mnum, "kmeans", n_clusters=n_clusters, plot_keys=plot_keys_drdr, drdr_reduced=drdr_reduced)

    if dbscan_clustering_flag:
        if run_local: print("----- DBSCAN CLUSTERING -----")
        if run_ppp: predicted = spectral_dbscan_clustering(data_ppp, plots_folder_ppp, mnum, "dbscan",  n_clusters=n_clusters, plot_keys=plot_keys_ppp, drdr_reduced=False)
        if run_drdr: predicted = spectral_dbscan_clustering(data_drdr, plots_folder_drdr, mnum, "dbscan", n_clusters=n_clusters, plot_keys=plot_keys_drdr, drdr_reduced=drdr_reduced)

    if plot_ground_truth_drdr_flag:
        if run_local: print("----- PRE-DEFINED CLUSTERS -----")
        plot_drdr_scatter_plot_classified(data_drdr, plots_folder_drdr, n_clusters=2, plot_keys=plot_keys_drdr)

    if bic_flag:
        if run_local: print("----- BIC CALCULATION -----")
        bic = bic_analysis_clustering(data_drdr, 15, results_folder_drdr)
        bic = bic_analysis_clustering(data_ppp, 15, results_folder_ppp)