import os
import sys

import numpy as np
import pandas as pd


def extract_participants_list(file_path):
    df = pd.read_excel(file_path)
    participants = df["IDs"]

    return participants


def extract_updrs_ppp_database(file_path):
    df = pd.read_excel(file_path)
    return df


def extract_updrs_dictionary_keys(path_to_updrs_keys):
    df = pd.read_excel(path_to_updrs_keys)
    columns_to_keep = ["FinalColumnsNames", "Description", "PPP_OFF_key", "PPP_ON_key", "PPP_JSON_number", "PPP_handness_relative_fields"]
    updrs_keys = df[columns_to_keep][0:34]

    return updrs_keys


def extract_ledd(path_to_ledd_file, participants):
    df = pd.read_csv(path_to_ledd_file)
    df = df[["pseudonym", "Timepoint", "LEDD"]]
    df = df[df["pseudonym"].isin(participants)]

    visits_folders = ["ses-POMVisit1", "ses-POMVisit2", "ses-POMVisit3"]
    df = df[df["Timepoint"].isin(visits_folders)]

    return df


def sum_with_nan_if_any(row):
    if row.isnull().any():
        return np.nan
    else:
        return row.sum()


def sum_keys(database, keys, column_name, averageFlag = False):
    for sheet_name, data in database.items():
        filteredData = data[keys]
        if averageFlag:
            data[column_name] = filteredData.apply(sum_with_nan_if_any, axis=1) / len(keys)
        else:
            data[column_name] = filteredData.apply(sum_with_nan_if_any, axis=1)
        database[sheet_name] = data
    return database


def safe_convert_to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def create_ledd_column(database, ledd, sheets, path_to_ppp_data_comp):
    for index, row in ledd.iterrows():
        if row["Timepoint"] == "ses-POMVisit1":
            database[sheets[0]].loc[database[sheets[0]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
            database[sheets[1]].loc[database[sheets[1]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
        elif row["Timepoint"] == "ses-POMVisit2":
            database[sheets[2]].loc[database[sheets[2]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
            database[sheets[3]].loc[database[sheets[3]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
        elif row["Timepoint"] == "ses-POMVisit3":
            database[sheets[4]].loc[database[sheets[4]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
            database[sheets[5]].loc[database[sheets[5]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])

    with pd.ExcelWriter(path_to_ppp_data_comp, engine='openpyxl') as writer:
        for sheet_name, data in database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    return database


if __name__ == "__main__":
    run_local = 1

    # Define base directories paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    path_to_participants_list = os.path.join(base_PPP_folder, "PPP_cohort_participants-with-tremor-list.xlsx")
    path_to_updrs_keys = os.path.join(base_PPP_folder, "updrs_analysis", "updrs_keys.xlsx")
    path_to_updrs_ppp_database = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_raw_data.xlsx")
    path_to_sub_dirs = "/project/3022026.01/pep/ClinVars_10-08-2023/"
    path_to_ledd_file = "/project/3022026.01/pep/ClinVars_10-08-2023/derivatives/merged_manipulated_2023-10-18.csv"
    path_to_ppp_data_comp = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_compmetrics.xlsx")

    if run_local: print("----- LOADING PARTICIPANTS LIST -----")
    participants = extract_participants_list(path_to_participants_list)

    if run_local: print("----- LOADING UPDRS PPP(POM) DATABASE -----")
    excel_file = pd.ExcelFile(path_to_updrs_ppp_database)
    sheets = excel_file.sheet_names
    ppp_database = {}
    for sheet in sheets:
        ppp_database[sheet] = pd.read_excel(path_to_updrs_ppp_database, sheet_name=sheet)
    # ppp_database = extract_updrs_ppp_database(path_to_updrs_ppp_database)

    if run_local: print("----- COMPUTING ADDITIONAL UPDRS METRICS -----")
    updrs_keys = extract_updrs_dictionary_keys(path_to_updrs_keys)

    ledd_exists = False
    if ledd_exists == False:
        levEqDose = extract_ledd(path_to_ledd_file, participants)
        ppp_database = create_ledd_column(ppp_database, levEqDose, sheets, path_to_ppp_data_comp)

    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][0:33], "TotalU3")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][0:33], "AvgTotalU3", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][23:33], "TremorUPDRS")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][23:33], "AvgTremorUPDRS", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][7:17], "Brady5Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][7:17], "AvgBrady5Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][7:20], "Brady14Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][np.concatenate([np.arange(7, 20), [22]])], "AvgBrady14Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, [updrs_keys["FinalColumnsNames"][i] for i in [27, 28, 29, 30, 32]], "LimbsRestTrem")
    ppp_database = sum_keys(ppp_database, [updrs_keys["FinalColumnsNames"][i] for i in [27, 28, 29, 30, 32]], "AvgLimbsRestTrem", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][3:7], "LimbsRigidity4Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][3:7], "AvgLimbsRigidity4Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][2:7], "LimbsRigidity")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][2:7], "AvgLimbsRigidity", averageFlag=True)
    ppp_database = sum_keys(ppp_database, ["AvgBrady5Items", "AvgLimbsRigidity"], "AvgLimbsBradyRig", averageFlag=True)
    ppp_database = sum_keys(ppp_database, ["Brady5Items", "LimbsRigidity"], "LimbsBradyRig")

    """
    The metrics that I defined based on Zach's paper are:
    - Brady14Items
    - AvgBrady14Items
    - LimbsRestTrem
    - AvgLimbsRestTrem
    - LimbsRigidity
    - AvgLimbsRigidity
    The others are scores computed as 
    """

    with pd.ExcelWriter(path_to_ppp_data_comp, engine='openpyxl') as writer:
        for sheet_name, data in ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print(f"----- DATASET WITH ADDITIONAL METRICS SAVED TO {path_to_ppp_data_comp} -----")
    sys.exit()
