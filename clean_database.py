import copy
import os
import sys
import pandas as pd

"""
This code cleans the database by removing all participants that had missing information. It creates two additional 
versions of the dataset. One version were the participants with NANs in the UPDRS scores are excluded, and one
database that includes only the participants that had complete information in the three visits.
"""


def clean_nan_rows(database, columns):
    for sheet_name, data in database.items():
        filtered_df = data[data[columns].notnull().all(1)]
        database[sheet_name] = filtered_df
    return database


def reduce_db_to_consistent_subjects(database, sheets):
    database_onoff = copy.deepcopy(database)

    # for sheet_name, data in database.items():
        # Delete any row that has at least one nan
        # filtered_df = data[data[:].notnull().all(1)]
        # filtered_df = data[data["LEDD"].notnull()]
        # database[sheet_name] = filtered_df
        # Make a deep copy

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

    common_ids = set.intersection(*sets_of_ids)
    common_ids_on = set.intersection(*sets_of_ids_on)
    common_ids_off = set.intersection(*sets_of_ids_off)

    for sheet_name, data in database.items():
        filtered_dfs = data[data['Subject'].isin(common_ids)]
        database[sheet_name] = filtered_dfs
        if "ON" in sheet_name:
            filtered_dfs_onoff = database_onoff[sheet_name][database_onoff[sheet_name]['Subject'].isin(common_ids_on)]
            database_onoff[sheet_name] = filtered_dfs_onoff
        elif "OFF" in sheet_name:
            filtered_dfs_onoff = database_onoff[sheet_name][database_onoff[sheet_name]['Subject'].isin(common_ids_off)]
            database_onoff[sheet_name] = filtered_dfs_onoff

    return database, database_onoff


def extract_updrs_dictionary_keys(path_to_updrs_keys):
    df = pd.read_excel(path_to_updrs_keys)
    columns_to_keep = ["FinalColumnsNames"]
    updrs_keys = df[columns_to_keep][0:49]

    return updrs_keys


if __name__ == "__main__":
    run_local = 1

    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    path_to_updrs_keys = os.path.join(base_PPP_folder, "updrs_analysis", "updrs_keys.xlsx")
    path_to_ppp_data_comp = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_compmetrics.xlsx")
    path_to_ppp_data_clean = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_cleanUPDRS.xlsx")
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_clean_and_reduced.xlsx")
    path_to_ppp_data_reduced_offon = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_clean_and_reduced_off-on.xlsx")

    if run_local: print("----- LOADING DATABASE -----")
    excel_file = pd.ExcelFile(path_to_ppp_data_comp)
    sheets = excel_file.sheet_names
    ppp_database = {}
    ppp_database_org = {}
    for sheet in sheets:
        ppp_database_org[sheet] = pd.read_excel(path_to_ppp_data_comp, sheet_name=sheet)
        ppp_database[sheet] = ppp_database_org[sheet].copy(deep=True)

    if run_local: print("----- DELETING ROWS WITH NANS IN THE UPDRS SCORES -----")
    updrs_keys = extract_updrs_dictionary_keys(path_to_updrs_keys)
    ppp_database = clean_nan_rows(ppp_database, updrs_keys["FinalColumnsNames"][0:33])

    if run_local: print(f"----- DATABASE SAVED TO {path_to_ppp_data_clean} -----")
    with pd.ExcelWriter(path_to_ppp_data_clean, engine='openpyxl') as writer:
        for sheet_name, data in ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print("----- INCLUDING ONLY PARTICIPANTS WITH COMPLETE DATA ACROSS THE 3 SESSIONS -----")
    [ppp_database, database_onoff] = reduce_db_to_consistent_subjects(ppp_database, sheets)

    if run_local: print(F"----- REDUCED DATABASE SAVED TO {path_to_ppp_data_reduced} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced, engine='openpyxl') as writer:
        for sheet_name, data in ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print(F"----- REDUCED OFF-ON DATABASE SAVED TO {path_to_ppp_data_reduced_offon} -----")
    with pd.ExcelWriter(path_to_ppp_data_reduced_offon, engine='openpyxl') as writer:
        for sheet_name, data in database_onoff.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    sys.exit()