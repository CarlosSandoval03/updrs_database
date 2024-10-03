import copy
import os
import sys
import pandas as pd

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from utilitiespkg import DataHandling

"""
This code cleans the database by removing all participants that had missing information. It creates two additional 
versions of the dataset. One version were the participants with NANs in the UPDRS scores are excluded, and one
database that includes only the participants that had complete information in the three visits.
"""
def clean_nan_rows(dataO, columns, path_to_subjects):
    database = copy.deepcopy(dataO)
    database = exclude_wrong_diagnose(database, path_to_subjects)
    for sheet_name, data in database.items():
        filtered_df = data[data[columns].notnull().all(axis=1)]
        filtered_df = filtered_df.reset_index(drop=True)
        database[sheet_name] = filtered_df
    return database


def exclude_wrong_diagnose(dataO, path_to_subjects):
    subjects = pd.read_csv(path_to_subjects)
    subjects = subjects["pseudonym"]
    database = copy.deepcopy(dataO)
    for sheet_name, data in database.items():
        filtered_df = data[~data["Subject"].isin(subjects)]
        filtered_df = filtered_df.reset_index(drop=True)
        database[sheet_name] = filtered_df
    return database


def reduce_db_to_consistent_subjects(dataO, sheets):
    # Deep copy the original data
    database = copy.deepcopy(dataO)
    database_onoff = copy.deepcopy(dataO)
    database_visits = copy.deepcopy(dataO)

    # Define sets of subject IDs for different categories
    sets_of_ids = [
        set(database[sheets[0]]['Subject']),
        set(database[sheets[1]]['Subject']),
        set(database[sheets[2]]['Subject']),
        set(database[sheets[3]]['Subject']),
        set(database[sheets[4]]['Subject']),
        set(database[sheets[5]]['Subject']),
    ]

    sets_of_ids_on = [
        set(database[sheets[1]]['Subject']),
        set(database[sheets[3]]['Subject']),
        set(database[sheets[5]]['Subject']),
    ]

    sets_of_ids_off = [
        set(database[sheets[0]]['Subject']),
        set(database[sheets[2]]['Subject']),
        set(database[sheets[4]]['Subject']),
    ]

    sets_of_ids_v1 = [
        set(database[sheets[0]]['Subject']),
        set(database[sheets[1]]['Subject']),
    ]

    sets_of_ids_v2 = [
        set(database[sheets[2]]['Subject']),
        set(database[sheets[3]]['Subject']),
    ]

    sets_of_ids_v3 = [
        set(database[sheets[4]]['Subject']),
        set(database[sheets[5]]['Subject']),
    ]

    # Find common subject IDs
    common_ids = set.intersection(*sets_of_ids)
    common_ids_on = set.intersection(*sets_of_ids_on)
    common_ids_off = set.intersection(*sets_of_ids_off)
    common_ids_v1 = set.intersection(*sets_of_ids_v1)
    common_ids_v2 = set.intersection(*sets_of_ids_v2)
    common_ids_v3 = set.intersection(*sets_of_ids_v3)

    # Filter and reset index for the main database
    for sheet_name, data in database.items():
        filtered_dfs = data[data['Subject'].isin(common_ids)].reset_index(drop=True)
        database[sheet_name] = filtered_dfs

        if "ON" in sheet_name:
            filtered_dfs_onoff = database_onoff[sheet_name][database_onoff[sheet_name]['Subject'].isin(common_ids_on)].reset_index(drop=True)
            database_onoff[sheet_name] = filtered_dfs_onoff
        elif "OFF" in sheet_name:
            filtered_dfs_onoff = database_onoff[sheet_name][database_onoff[sheet_name]['Subject'].isin(common_ids_off)].reset_index(drop=True)
            database_onoff[sheet_name] = filtered_dfs_onoff

    # Filter and reset index for the visits database
    for sheet_name, data in database_visits.items():
        if "Visit 1" in sheet_name:
            filtered_dfs = database_visits[sheet_name][database_visits[sheet_name]['Subject'].isin(common_ids_v1)].reset_index(drop=True)
        elif "Visit 2" in sheet_name:
            filtered_dfs = database_visits[sheet_name][database_visits[sheet_name]['Subject'].isin(common_ids_v2)].reset_index(drop=True)
        elif "Visit 3" in sheet_name:
            filtered_dfs = database_visits[sheet_name][database_visits[sheet_name]['Subject'].isin(common_ids_v3)].reset_index(drop=True)
        
        database_visits[sheet_name] = filtered_dfs

    return database, database_onoff, database_visits


def extract_updrs_dictionary_keys(path_to_updrs_keys):
    df = pd.read_excel(path_to_updrs_keys)
    columns_to_keep = ["FinalColumnsNames"]
    updrs_keys = df[columns_to_keep][0:49]

    return updrs_keys


def filterHandsTremorSubjects(eval_keys, threshold, *dfs):
    dataframes = [copy.deepcopy(df) for df in dfs]

    indices = []
    for df in dataframes:
        idxRUE = df[eval_keys[0]] >= threshold
        idxLUE = df[eval_keys[1]] >= threshold
        indices.append((idxRUE | idxLUE))

    combined_indices = indices[0]
    for idx in indices[1:]:
        combined_indices = combined_indices & idx

    result_indexes = combined_indices[combined_indices].index.tolist()

    return result_indexes


def extract_participants_list(file_path):
    df = pd.read_excel(file_path)
    participants = df["Subject"]

    return participants


if __name__ == "__main__":
    run_local = 1
    ignore_unmedicated_participants_flag = True

    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    path_to_updrs_keys = os.path.join(base_PPP_folder, "updrs_keys.xlsx")
    path_to_ppp_data_comp = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_compmetrics.xlsx")
    path_to_ppp_data_clean = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_cleanUPDRS.xlsx")
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")
    path_to_ppp_data_reduced_offon = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects_by_off-on.xlsx")
    path_to_ppp_data_reduced_visits = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects_by_visit.xlsx")

    path_to_ppp_data_reduced_f = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects_trem-filtered.xlsx")
    path_to_ppp_data_reduced_offon_f = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects_by_off-on_trem-filtered.xlsx")
    path_to_ppp_data_reduced_visits_f = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects_by_visit_trem-filtered.xlsx")

    path_to_excluded_subjects = "/project/3024023.01/PPP-POM_cohort/exclusions_diagnosis.csv"
    path_to_include_subjects = "/project/3024023.01/PPP-POM_cohort/PPP_cohort_participants-with-tremor-list.xlsx"

    if run_local: print("----- LOADING DATABASE -----")
    excel_file = pd.ExcelFile(path_to_ppp_data_comp)
    sheets = excel_file.sheet_names
    ppp_database = {}
    ppp_database_org = {}
    for sheet in sheets:
        ppp_database_org[sheet] = pd.read_excel(path_to_ppp_data_comp, sheet_name=sheet)
        ppp_database[sheet] = ppp_database_org[sheet].copy(deep=True)

    if run_local: print("----- LOADING LIST OF PARTICIPANTS WITH TREMOR DURING SCANS -----")
    participants_to_include = extract_participants_list(path_to_include_subjects)

    if run_local: print("----- DELETING ROWS WITH NANS IN THE UPDRS SCORES -----")
    updrs_keys = extract_updrs_dictionary_keys(path_to_updrs_keys)
    ppp_database = clean_nan_rows(ppp_database, updrs_keys["FinalColumnsNames"][0:33], path_to_excluded_subjects)

    if run_local: print("----- DELETING ROWS WITH NANS AND 0s IN THE LED SCORES -----")
    if ignore_unmedicated_participants_flag:
        objDataHandling = DataHandling(ppp_database, "ppp")
        ppp_database = objDataHandling.remove_unmedicated_subjects()

    if run_local: print(f"----- DATABASE SAVED TO {path_to_ppp_data_clean} -----")
    with pd.ExcelWriter(path_to_ppp_data_clean, engine='openpyxl') as writer:
        for sheet_name, data in ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print("----- INCLUDING ONLY PARTICIPANTS WITH COMPLETE DATA ACROSS THE 3 SESSIONS -----")
    [ppp_database, database_onoff, database_visits] = reduce_db_to_consistent_subjects(ppp_database, sheets)

    if run_local: print(F"----- REDUCED DATABASE SAVED TO {path_to_ppp_data_reduced} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced, engine='openpyxl') as writer:
        for sheet_name, data in ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print(F"----- REDUCED OFF-ON DATABASE SAVED TO {path_to_ppp_data_reduced_offon} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced_offon, engine='openpyxl') as writer:
        for sheet_name, data in database_onoff.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            
    if run_local: print(F"----- REDUCED VISITS DATABASE SAVED TO {path_to_ppp_data_reduced_visits} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced_visits, engine='openpyxl') as writer:
        for sheet_name, data in database_visits.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    tremor_threshold = 1
    idx1 = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], tremor_threshold, database_visits[sheets[0]])
    idx1 = set(database_visits[sheets[0]]['Subject'].loc[idx1].reset_index(drop=True))
    idx2 = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], tremor_threshold, database_visits[sheets[2]])
    idx2 = set(database_visits[sheets[2]]['Subject'].loc[idx2].reset_index(drop=True))
    idx3 = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], tremor_threshold, database_visits[sheets[4]])
    idx3 = set(database_visits[sheets[4]]['Subject'].loc[idx3].reset_index(drop=True))
    # idx2.update(idx1)
    # idx3.update(idx2)
    idx = [idx1, idx1, idx2, idx2, idx3, idx3]
    filtered_database_visits = dict()
    for i, sheet in enumerate(sheets):
        filtered_database_visits[sheet] = ppp_database[sheet][ppp_database[sheet]["Subject"].isin(idx[i])].reset_index(drop=True)
    # filtered_database_visits = {sheet_name: data.loc[idx[i]].reset_index(drop=True) for i, (sheet_name, data) in enumerate(database_visits.items())}

    # idx_all = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], tremor_threshold, ppp_database[sheets[0]], ppp_database[sheets[2]], ppp_database[sheets[4]])
    idx_all = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], tremor_threshold, ppp_database[sheets[0]])    # Consider only tremor at first session
    select_sub = ppp_database[sheets[0]]['Subject'][idx_all]
    filtered_ppp_database = dict()
    for sheet in sheets:
        # filtered_ppp_database[sheet] = ppp_database[sheet][ppp_database[sheet]["Subject"].isin(select_sub) | ppp_database[sheet]["Subject"].isin(participants_to_include)].reset_index(drop=True)
        filtered_ppp_database[sheet] = ppp_database[sheet][
            (ppp_database[sheet]["Subject"].isin(participants_to_include)) |
            # (ppp_database[sheet]["Subject"].isin(participants_to_include) & ppp_database[sheet]["Subject"].isin(idx2)) |
            # (ppp_database[sheet]["Subject"].isin(participants_to_include) & ppp_database[sheet]["Subject"].isin(idx3)) |
            (ppp_database[sheet]["Subject"].isin(select_sub) & ppp_database[sheet]["Subject"].isin(idx2)) |
            (ppp_database[sheet]["Subject"].isin(select_sub) & ppp_database[sheet]["Subject"].isin(idx3))
            ].reset_index(drop=True)

    if run_local: print(F"----- REDUCED DATABASE SAVED TO {path_to_ppp_data_reduced_f} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced_f, engine='openpyxl') as writer:
        for sheet_name, data in filtered_ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    # Saving the filtered database_visits
    if run_local: print(F"----- REDUCED VISITS DATABASE SAVED TO {path_to_ppp_data_reduced_visits_f} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced_visits_f, engine='openpyxl') as writer:
        for sheet_name, data in filtered_database_visits.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    sys.exit()