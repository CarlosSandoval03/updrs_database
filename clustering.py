import os
import sys
import copy
from pathlib import Path

import nibabel as nib
import scipy as spy
import glob
import json
import scipy.io
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats
from scipy.stats import t
from scipy.special import gammaln
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.ticker import PercentFormatter
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot


def spectral_dbscan_clustering(data, results_folder, clustering_type="spectral", n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"]):
    model = {
        "spectral": SpectralClustering(n_clusters=n_clusters),
        "dbscan":  DBSCAN(eps=0.30, min_samples=9)
    }.get(clustering_type)
    # model = SpectralClustering(n_clusters=n_clusters)
    yhat = model.fit_predict(data)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plotx = [data[plot_keys[0]][i] for i in row_ix]
        ploty = [data[plot_keys[1]][i] for i in row_ix]
        pyplot.scatter(plotx, ploty)
    figure_name = os.path.join(results_folder, f"clustering_{clustering_type}.png")
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return 0


def general_model_clustering(data, results_folder, clustering_type="gaussian", n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"]):
    model = {
        "gaussian": GaussianMixture(n_components=n_clusters),
        "kmeans": KMeans(n_clusters=n_clusters)
    }.get(clustering_type)

    model.fit(data)
    yhat = model.predict(data)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plotx = [data[plot_keys[0]][i] for i in row_ix]
        ploty = [data[plot_keys[1]][i] for i in row_ix]
        pyplot.scatter(plotx, ploty)
    figure_name = os.path.join(results_folder, f"clustering_{clustering_type}.png")
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return 0


if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")

    base_drdr_folder = "/project/3024023.01/fMRI_DRDR/"
    results_folder_drdr = "/project/3024023.01/fMRI_DRDR/updrs_analysis/"
    path_to_drdr_data = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean.xlsx")

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
    spectral_clustering_flag = True
    k_means_clustering_flag = True
    gaussian_clustering_flag = True
    dbscan_clustering_flag = True

    keys_ppp = ["FreqPeak", "LogPower", "TremorUPDRS", "PosturalTremor", "Brady14Items", "LimbsRestTrem", "LimbsRigidity5Items"]
    keys_drdr = ["OFFTremorUPDRS", "AvgBrady14Items", "AvgLimbsRestTrem", "AvgLimbsRigidity"]
    data_ppp = ppp_database[sheets_ppp[0]][keys_ppp]
    data_drdr = drdr_database[sheets_drdr[0]][keys_drdr]

    run_ppp = True
    run_drdr = True

    plots_folder_ppp = os.path.join(results_folder_ppp, "clustering")
    plots_folder_drdr = os.path.join(results_folder_drdr, "clustering")
    Path(plots_folder_drdr).mkdir(parents=True, exist_ok=True)
    Path(plots_folder_ppp).mkdir(parents=True, exist_ok=True)
    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if spectral_clustering_flag:
        if run_local: print("----- SPECTRAL CLUSTERING -----")
        if run_ppp: spectral_dbscan_clustering(data_ppp, plots_folder_ppp, "spectral",  n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"])
        if run_drdr: spectral_dbscan_clustering(data_drdr, plots_folder_drdr, "spectral", n_clusters=3, plot_keys=["AvgLimbsRigidity", "AvgLimbsRestTrem"])

    if gaussian_clustering_flag:
        if run_local: print("----- GAUSSIAN MIXTURE MODEL CLUSTERING -----")
        if run_ppp: general_model_clustering(data_ppp, plots_folder_ppp, "gaussian", n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"])
        if run_drdr: general_model_clustering(data_drdr, plots_folder_drdr, "gaussian", n_clusters=3, plot_keys=["AvgLimbsRigidity", "AvgLimbsRestTrem"])

    if k_means_clustering_flag:
        if run_local: print("----- KMEANS CLUSTERING -----")
        if run_ppp: general_model_clustering(data_ppp, plots_folder_ppp, "kmeans", n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"])
        if run_drdr: general_model_clustering(data_drdr, plots_folder_drdr, "kmeans", n_clusters=3, plot_keys=["AvgLimbsRigidity", "AvgLimbsRestTrem"])

    if dbscan_clustering_flag:
        if run_local: print("----- DBSCAN CLUSTERING -----")
        if run_ppp: spectral_dbscan_clustering(data_ppp, plots_folder_ppp, "dbscan",  n_clusters=3, plot_keys=["LogPower", "LimbsRestTrem"])
        if run_drdr: spectral_dbscan_clustering(data_drdr, plots_folder_drdr, "dbscan", n_clusters=3, plot_keys=["AvgLimbsRigidity", "AvgLimbsRestTrem"])

