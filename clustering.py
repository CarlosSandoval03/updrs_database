import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from bicpkg import BICAnalysis
from clusteringpkg import Clustering


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

    figure_name = os.path.join(results_folder, f"pre-clustering_DRDR_{model_name}_clusters={n_clusters}.png")
    plt.xlabel(plot_keys[0])
    plt.ylabel(plot_keys[1])
    plt.title(f"Predefined DRDR Clustering.")
    plt.legend()
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return 0



if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    # path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")
    path_to_ppp_data_reduced = os.path.join(results_folder_ppp, "ppp_updrs_trem-filtered_database_of_predictors.xlsx")

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
    k_means_clustering_flag = False
    gaussian_clustering_flag = False
    plot_clusters_plots_flag = True
    plot_ground_truth_drdr_flag = False
    bic_flag_drdr = False
    bic_analysis_flag = False

    # predictors_ppp = ["FreqPeak", "LogPower", "TremorUPDRS", "Brady14Items", "LimbsRestTrem", "LimbsRigidity5Items"]
    # predictors_ppp = ["ChangeTremorUPDRS_1", "ChangeLimbsRestTrem_1", "ChangeTremorUPDRS_2", "ChangeLimbsRestTrem_2", "ChangeTremorUPDRS_3", "ChangeLimbsRestTrem_3"]
    predictors_ppp = ["ChangeTremorUPDRS_3", "ChangeLimbsRestTrem_3"]
    # predictors_ppp = ["ResponseRestTrem_1", "ResponseRestTrem_2", "ResponseRestTrem_3"]

    # predictors_drdr = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem", "ResponseTotalU3", "ResponseBrady14Items", "ResponseLimbsRigidity5Items"]
    predictors_drdr = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]

    run_ppp = True
    run_drdr = False

    plots_folder_drdr = os.path.join(results_folder_drdr, "clustering")
    Path(plots_folder_drdr).mkdir(parents=True, exist_ok=True)

    # model_name = "Model_12"
    model_name = "Model_Change-Trem-RestTrem_Year2"
    n_clusters = 2
    drdr_reduced = True # Whether I consider only 34 sub or all of them

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    print("----- CREATING CLUSTERING CLASS OBJECTS -----")
    objClustPPP = Clustering(ppp_database, "ppp")
    objClustDRDR = Clustering(drdr_database, "drdr")

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if spectral_clustering_flag:
        if run_local: print("----- SPECTRAL CLUSTERING -----")
        if run_ppp: objClustPPP.apply_clustering(clust_type="spectral", n_clusters=n_clusters, predictors=predictors_ppp, model_name=model_name)
        if run_drdr: objClustDRDR.apply_clustering(clust_type="spectral", n_clusters=n_clusters, predictors=predictors_ppp, model_name=model_name)

    if gaussian_clustering_flag:
        if run_local: print("----- GAUSSIAN MIXTURE MODEL CLUSTERING -----")
        if run_ppp: objClustPPP.apply_clustering(clust_type="gaussian", n_clusters=n_clusters, predictors=predictors_ppp, model_name=model_name)
        if run_drdr: objClustDRDR.apply_clustering(clust_type="gaussian", n_clusters=n_clusters, predictors=predictors_ppp, model_name=model_name)

    if k_means_clustering_flag:
        if run_local: print("----- KMEANS CLUSTERING -----")
        if run_ppp: objClustPPP.apply_clustering(clust_type="kmeans", n_clusters=n_clusters, predictors=predictors_ppp, model_name=model_name)
        if run_drdr: objClustDRDR.apply_clustering(clust_type="kmeans", n_clusters=n_clusters, predictors=predictors_ppp, model_name=model_name)

    if plot_ground_truth_drdr_flag:
        if run_local: print("----- PRE-DEFINED CLUSTERS -----")
        data_drdr = drdr_database[sheets_drdr[0]][predictors_drdr]
        plot_drdr_scatter_plot_classified(data_drdr, plots_folder_drdr, n_clusters=n_clusters, plot_keys=[predictors_drdr[0], predictors_drdr[1]])

    if bic_flag_drdr:
        if run_local: print("----- BIC CALCULATION -----")
        add_name = "bic_by_num-clusters_k-means_clustering"

        data_drdr = drdr_database[sheets_drdr[0]][predictors_drdr]
        bic_obj = BICAnalysis(data_drdr)
        bic = bic_obj.create_line_plots(15, results_folder_drdr, add_name)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if bic_analysis_flag:
        if run_local: print("----- RUNNING BIC ANALYSIS -----")
        results_folder = results_folder_ppp
        # predictors_ppp = ["ChangeTremorUPDRS_1", "ChangeLimbsRestTrem_1", "ChangeTremorUPDRS_2", "ChangeLimbsRestTrem_2", "ChangeTremorUPDRS_3", "ChangeLimbsRestTrem_3"]
        predictors_ppp = ["ChangeTremorUPDRS_3", "ChangeLimbsRestTrem_3"]
        # predictors_ppp = ["ResponseRestTrem_1", "ResponseRestTrem_2", "ResponseRestTrem_3"]

        # add_name = "bic_k-means_Model12"
        add_name = "bic_k-means_Model_Change-Trem-RestTrem_Year2"
        data_ppp = ppp_database[sheets_ppp[0]][predictors_ppp]

        bic_obj = BICAnalysis(data_ppp)
        bic = bic_obj.create_line_plots(15, results_folder, add_name)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_clusters_plots_flag:
        plotting_keys = ["ChangeTremorUPDRS", "ChangeLimbsRestTrem"]
        objClustPPP.clusters_scatter_plot(plotting_keys=plotting_keys, clustering_method="two-steps", visit_name="Baseline")
        objClustPPP.clusters_scatter_plot(plotting_keys=plotting_keys, clustering_method="two-steps", visit_name="Baseline", model_name="Labels_Model12")
        objClustPPP.clusters_scatter_plot(plotting_keys=plotting_keys, clustering_method="two-steps", visit_name="Baseline", model_name="Labels_Model11b")

        objClustPPP.clusters_scatter_plot(plotting_keys=plotting_keys, clustering_method="arbitrary-updrs", visit_name="Baseline")

        path_to_extra_labels = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/Clustering/kmeans_Model_12_clusters=2_.csv"
        objClustPPP.clusters_scatter_plot(plotting_keys=plotting_keys, clustering_method="other", visit_name="Baseline", path_to_labels=path_to_extra_labels)