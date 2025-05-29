import copy
import os
import shutil
import sys
import pandas as pd

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from utilitiespkg import SearchDirectories


if __name__ == "__main__":
    run_local = 1
    ignore_unmedicated_participants_flag = True

    path_to_list_1 = "/project/3024023.01/PPP-POM_cohort/PPP_cohort_participants-with-tremor-list.xlsx"
    path_to_list_2 = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/ppp_updrs_trem-filtered_database_of_predictors.xlsx"

    path_to_participants_result = "/project/3024023.01/PPP-POM_cohort/fmriprep_missing.xlsx"

    searchObj = SearchDirectories()

    if run_local: print("----- LOADING LIST OF PARTICIPANTS WITH TREMOR DURING SCANS -----")
    participants1 = searchObj.extract_participants_list(path_to_list_1)
    participants2 = searchObj.extract_participants_list(path_to_list_2)

    list_subjects = participants2[~participants2.isin(participants1)]

    df = pd.DataFrame()
    df["Subject"] = pd.Series(list_subjects).reset_index(drop=True)

    df.to_excel(path_to_participants_result, index=False)
