import os
import sys
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg'
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Agg'

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from plottingpkg import UPDRSPlotting


def sort_visits_and_groups(df):
    # Define custom order for 'Visit' and 'Group'
    visit_order = ['Baseline', 'Year 1', 'Year 2']
    group_order = ['Resistant', 'Intermediate', 'Responsive']

    # Convert columns to categorical with defined order
    df['Visit'] = pd.Categorical(df['Visit'], categories=visit_order, ordered=True)
    df['Group'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)

    # Sort by Visit first, then Group
    return df.sort_values(by=['Visit', 'Group']).reset_index(drop=True)

if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis_154/"
    path_to_ppp_data_reduced = os.path.join(results_folder_ppp, "ppp_updrs_database_consistent_subjects_trem-filtered.xlsx")
    filtered_string = "" # "TremFiltered"
    path_to_consistent_sub_responsiveness_profile = os.path.join(results_folder_ppp, f"{filtered_string}Dopamine_responsiveness_profile.xlsx")

    base_results_folder = f"/project/3024023.01/Article stuff/"
    base_drdr_folder = "/project/3024023.01/fMRI_DRDR/"
    results_folder_drdr = "/project/3024023.01/fMRI_DRDR/updrs_analysis/"
    path_to_drdr_data = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean_complete.xlsx")

    valid_subjects_drdr = ["sub-p30", "sub-p08", "sub-p11", "sub-p28", "sub-p27", "sub-p42", "sub-p50", "sub-p72", "sub-p75",
                           "sub-p74", "sub-p73", "sub-p78", "sub-p81", "sub-p83", "sub-p18", "sub-p02", "sub-p60", "sub-p59",
                           "sub-p38", "sub-p49", "sub-p40", "sub-p19", "sub-p29", "sub-p36", "sub-p33", "sub-p71", "sub-p21",
                           "sub-p70", "sub-p64", "sub-p56", "sub-p48", "sub-p43", "sub-p76", "sub-p77"]

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

    for key in drdr_database:
        df = drdr_database[key]
        drdr_database[key] = df[df['Subject'].isin(valid_subjects_drdr)]
        drdr_database[key] = drdr_database[key].reset_index(drop=True)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    objDataHandlingPPP = UPDRSPlotting(ppp_database, "ppp", trem_filtered=False, updrs_analysis_folder="updrs_analysis_154")
    objDataHandlingDRDR = UPDRSPlotting(drdr_database, "drdr", trem_filtered=False, updrs_analysis_folder="updrs_analysis")

    updrs_keys = ['LimbsRestTrem'] #["ChangeTotalU3", "ChangeTremorUPDRS", "ChangeLimbsRestTrem", "ChangeBrady14Items", "ChangeLimbsRigidity5Items"]
    confs_create_df = {
            'updrs': [
                # {'off': 'ChangeTotalU3', 'on': 'ChangeTotalU3'},
                # {'off': 'ChangeTremorUPDRS', 'on': 'ChangeTremorUPDRS'},
                # {'off': 'ChangeLimbsRestTrem', 'on': 'ChangeLimbsRestTrem'},
                {'off': 'LimbsRestTrem', 'on': 'LimbsRestTrem'}
                # {'off': 'ChangeBrady14Items', 'on': 'ChangeBrady14Items'},
                # {'off': 'ChangeLimbsRigidity5Items', 'on': 'ChangeLimbsRigidity5Items'},
            ],
            'subplots': [
                {'groups': ['Resistant', 'Intermediate', 'Responsive'], 'visits': ['Baseline', 'Year 1', 'Year 2'], 'sessions': ['off', 'on']},
            ],
        }

    dataframes_created_ppp = objDataHandlingPPP.obtain_formatted_dataframe(confs_create_df, "arbitrary-updrs", FilteredByHandsTremorFlag=False, path_to_labels="", model_labels="")

    for i, key in enumerate(updrs_keys):
        dataframes_created_ppp[i] = sort_visits_and_groups(dataframes_created_ppp[i])
        dataframes_created_ppp[i].to_csv(os.path.join(base_results_folder, f"df_ppp_{key}.csv"), index=False)

    updrs_keys = ['LimbsRestTrem'] #["ChangeTotalU3", "ChangeTremorUPDRS", "ChangeLimbsRestTrem", "ChangeBrady14Items", "ChangeLimbsRigidity5Items"]
    confs_create_df = {
            'updrs': [
                # {'off': 'ChangeTotalU3', 'on': 'ChangeTotalU3'},
                # {'off': 'ChangeTremorUPDRS', 'on': 'ChangeTremorUPDRS'},
                # {'off': 'ChangeLimbsRestTrem', 'on': 'ChangeLimbsRestTrem'},
                {'off': 'UPDRS_OFF_L_RT', 'on': 'UPDRS_ON_L_RT'}
                # {'off': 'ChangeBrady14Items', 'on': 'ChangeBrady14Items'},
                # {'off': 'ChangeLimbsRigidity5Items', 'on': 'ChangeLimbsRigidity5Items'},
            ],
            'subplots': [
                {'groups': ['Resistant', 'Intermediate', 'Responsive'], 'visits': ['Baseline'], 'sessions': ['off', 'on']},
            ],
        }
    dataframes_created_drdr = objDataHandlingDRDR.obtain_formatted_dataframe(confs_create_df, "arbitrary-updrs", FilteredByHandsTremorFlag=False, path_to_labels="", model_labels="")
    for i, key in enumerate(updrs_keys):
        dataframes_created_drdr[i].to_csv(os.path.join(base_results_folder, f"df_drdr_{key}.csv"), index=False)

    for i, dfi_ppp in enumerate(dataframes_created_ppp):
        dfi_ppp["Cohort"] = "ppp"
        dataframes_created_ppp[i] = dfi_ppp
        dataframes_created_drdr[i]["Cohort"] = "drdr"
        dataframes_created_drdr[i][updrs_keys[i]] = dataframes_created_drdr[i][updrs_keys[i]] * 100

        merged_df = pd.merge(dataframes_created_ppp[i], dataframes_created_drdr[i],
                              on=["Subject", "Group", "Visit", "Session", "Cohort", updrs_keys[i]], how="outer")
        merged_df.to_csv(os.path.join(base_results_folder, f"df_merged_{updrs_keys[i]}.csv"), index=False)
