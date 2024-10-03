import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy.spatial import distance
from sklearn import cluster
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from utilitiespkg import DataHandling
from statisticspkg import StatisticsHelper


if __name__ == "__main__":
    run_local = True

    # Paths
    filtered_string = "TremFiltered"
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects_trem-filtered.xlsx")
    path_to_consistent_sub_responsiveness_profile = os.path.join(base_PPP_folder, "updrs_analysis", f"{filtered_string}Dopamine_responsiveness_profile.xlsx")

    ignore_unmedicated_participants_flag = False

    general_stats_table_flag = False
    anova_off_on_flag = False
    mixed_effects_flag = True
    plot_clustering_grid_flag = False
    plot_clusters_plots_flag = False
    plot_clusters_bars_flag = False
    plot_clusters_stability_flag = False
    
    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if run_local: print("----- LOADING DATABASE PPP -----")
    excel_file_ppp = pd.ExcelFile(path_to_ppp_data_reduced)
    sheets_ppp = excel_file_ppp.sheet_names
    ppp_database = {}
    for sheet in sheets_ppp:
        ppp_database[sheet] = pd.read_excel(path_to_ppp_data_reduced, sheet_name=sheet)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    """ INITIALIZE OBJECTS OF RELEVENT CLASSES """
    objStatsPPP = StatisticsHelper(ppp_database, 'ppp')

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if ignore_unmedicated_participants_flag:
        ppp_database = objStatsPPP.remove_unmedicated_subjects(ppp_database)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if general_stats_table_flag:
        objStatsPPP.create_stats_table()

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if anova_off_on_flag:
        print(" ----- REPEATED MEASURES ANOVA ----- ")
        updrs_keys = ["RestTrem", "LimbsRestTrem", "Brady14Items", "LimbsRigidity5Items"]
        objStatsPPP.repeated_measures_anova(keys_to_analyze=updrs_keys)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if mixed_effects_flag:
        print(" ----- MIXED EFFECTS LM ----- ")
        updrs_keys = ["ResponseRestTrem", "ChangeLimbsRestTrem", "ChangeBrady14Items", "ChangeLimbsRigidity5Items"]
        model_name = "Model12"
        objStatsPPP.mixed_effects(keys_to_analyze=updrs_keys, classification_type="two-steps", model_name=model_name, use_covariates=True)
        objStatsPPP.mixed_effects(keys_to_analyze=updrs_keys, classification_type="two-steps", model_name="Model12_3clust", use_covariates=True, model_labels="Labels_Model12")
        objStatsPPP.mixed_effects(keys_to_analyze=updrs_keys, classification_type="two-steps", model_name="Model11", use_covariates=True, model_labels="Labels_Model11b")

        objStatsPPP.mixed_effects(keys_to_analyze=updrs_keys, classification_type="arbitrary-updrs", use_covariates=True)
        path_to_extra_labels = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/Clustering/kmeans_Model_12_clusters=2_.csv"
        objStatsPPP.mixed_effects(keys_to_analyze=updrs_keys, classification_type="other", path_to_labels=path_to_extra_labels, use_covariates=True)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_clustering_grid_flag:
        if run_local: print(" ----- PLOT COMPARISON BY CLUSTERING METHODS -----")
        methods_to_compare = ["two-steps", "arbitrary-updrs"]
        objStatsPPP.plot_subject_wise_cluster_comparison(methods_to_compare)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_clusters_bars_flag:
        if run_local: print("----- PLOT PARTICIPANTS CLUSTER PROGRESSION -----")
        objStatsPPP.plot_participant_clusters_bars()

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_clusters_stability_flag:
        if run_local: print("----- PLOT PARTICIPANTS CLUSTER STABILITY -----")
        objStatsPPP.plot_participant_clusters_stability()




    sys.exit()