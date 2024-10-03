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
    path_to_participants_list = os.path.join(base_PPP_folder, "Complete_ppp_participants_list.xlsx")
    path_to_bids_folder = "/project/3022026.01/pep/bids/"

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if run_local: print("----- LOADING DATABASE PPP -----")
    folders = [folder for folder in os.listdir(path_to_bids_folder) if
               folder.startswith("sub-POM") and os.path.isdir(os.path.join(path_to_bids_folder, folder))]

    subjects = dict()
    subjects["sub-IDs"] = pd.DataFrame()
    subjects["sub-IDs"]["IDs"] = folders

    if run_local: print("----- ADDING TWO-STEPS CLUSTERS LABELS -----")
    with pd.ExcelWriter(path_to_participants_list, engine='openpyxl') as writer:
        for sheet_name, data in subjects.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)