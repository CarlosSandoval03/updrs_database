import copy
import os
import shutil
import sys
import pandas as pd

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from utilitiespkg import SearchDirectories


def extract_participants_list(file_path):
    df = pd.read_excel(file_path)
    participants = df["Subject"]

    return participants


if __name__ == "__main__":
    run_local = 1
    ignore_unmedicated_participants_flag = True

    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    path_to_my_bids = os.path.join(base_PPP_folder, "bids")
    path_to_ppp_bids = "/project/3022026.01/pep/bids/"
    path_to_sub_handedness = os.path.join(base_PPP_folder, "bids", "subjects_handedness.xlsx")
    path_to_participants_list = os.path.join(base_PPP_folder, "bids", "existance_of_sessions.xlsx")
    path_to_complete_data = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_raw_data.xlsx")

    if run_local: print("----- LOADING LIST OF PARTICIPANTS WITH TREMOR DURING SCANS -----")
    participants = extract_participants_list(path_to_participants_list)

    excel_file = pd.ExcelFile(path_to_complete_data)
    sheets = excel_file.sheet_names
    ppp_database = pd.read_excel(path_to_complete_data, sheet_name=sheets[0])

    list_subjects = []
    list_handedness = []
    list_devicehand = []

    for subject in participants:
        list_subjects.append(subject)
        list_handedness.append(int(ppp_database["Handedness"][ppp_database["Subject"] == subject].iloc[0]))
        list_devicehand.append(int(ppp_database["DeviceHand"][ppp_database["Subject"] == subject].iloc[0]))

    handedness_summary = pd.DataFrame()
    handedness_summary["Subject"] = pd.Series(list_subjects).reset_index(drop=True)
    handedness_summary["Handedness"] = pd.Series(list_handedness).reset_index(drop=True)
    handedness_summary["DeviceHand"] = pd.Series(list_devicehand).reset_index(drop=True)

    handedness_summary.to_excel(path_to_sub_handedness, index=False)
