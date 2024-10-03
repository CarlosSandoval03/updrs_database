import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")
    path_to_labels = os.path.join(base_PPP_folder, "updrs_analysis/Clustering_labels_two-steps_baseline.xlsx")

    add_labels = True
    analysis_name_cluster_response = "arbitraryUPDRS"
    # analysis_name_cluster_response = "two-steps-Clustering"

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if run_local: print("----- LOADING DATABASE PPP -----")
    excel_file_ppp = pd.ExcelFile(path_to_ppp_data_reduced)
    sheets_ppp = excel_file_ppp.sheet_names
    ppp_database = {}
    for sheet in sheets_ppp:
        ppp_database[sheet] = pd.read_excel(path_to_ppp_data_reduced, sheet_name=sheet)

    print("----- LOAD LABELS -----")
    excel_file = pd.ExcelFile(path_to_labels)
    sheets = excel_file.sheet_names
    labels = {}
    labels = pd.read_excel(path_to_labels, sheet_name=sheets[0])
    ppp_database[sheets_ppp[0]] = df2_with_labels = pd.merge(ppp_database[sheets_ppp[0]], labels[['Subject', 'Cluster:1=Resp;2=Resi']], on='Subject', how='left')
    ppp_database[sheets_ppp[0]].rename(columns={'Cluster:1=Resp;2=Resi': 'Predict_Model4'}, inplace=True)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if add_labels:
        print("----- ADDING TWO-STEPS CLUSTERS LABELS -----")
        with pd.ExcelWriter(path_to_ppp_data_reduced, engine='openpyxl') as writer:
            for sheet_name, data in ppp_database.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False)

    sys.exit()