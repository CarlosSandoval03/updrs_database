import os
import sys

import numpy as np
import pandas as pd

import scipy.io as sio

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from utilitiespkg import DataHandling

filterHandsTremorSubjects = DataHandling.get_subjects_above_threshold
percentage_change_Elble = DataHandling.percentage_change_elble
percentage_change_Basic = DataHandling.percentage_change_basic

def extract_participants_list(file_path):
    df = pd.read_excel(file_path)
    participants = df["Subject"]

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


def compute_tremor_responsiveness(database, keys, column_name, sheets):
    # Computes the OFF - ON difference per visit.
    visits = [0, 1, 2]
    for i, key in enumerate(keys):
        for visit in visits:
            filteredData_off = database[sheets[2*visit]][key]
            weightsData_off = database[sheets[2 * visit]]["U18ConsResTrem"] / 4
            filteredData_on = database[sheets[2 * visit + 1]][key]
            weightsData_on = database[sheets[2 * visit + 1]]["U18ConsResTrem"] / 4

            weightsData_off = weightsData_off.replace(0, 0.25)
            weightsData_on = weightsData_on.replace(0, 0.25)

            responsiveness = (filteredData_off * weightsData_off) - (filteredData_on * weightsData_on)

            database[sheets[2 * visit]]["WRestTrem"] = filteredData_off * weightsData_off
            database[sheets[2 * visit + 1]]["WRestTrem"] = filteredData_on * weightsData_on
            database[sheets[2 * visit]][column_name[i]] = responsiveness
            database[sheets[2 * visit + 1]][column_name[i]] = responsiveness

    return database


def dopamine_responsiveness(database, keys, column_name, sheets, percentage_change_function=percentage_change_Elble):
    # Computes the OFF - ON difference per visit.
    visits = [0, 1, 2]
    for i, key in enumerate(keys):
        for visit in visits:
            filteredData_off = database[sheets[2*visit]][key]
            filteredData_on = database[sheets[2 * visit + 1]][key]

            responsiveness = percentage_change_function(filteredData_on, filteredData_off)

            database[sheets[2*visit]][column_name[i]] = responsiveness
            database[sheets[2*visit+1]][column_name[i]] = responsiveness

    return database


def safe_convert_to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def create_ledd_column(database, ledd, sheets, path_to_ppp_data_comp):
    for index, row in ledd.iterrows():
        if row["Timepoint"] == "ses-POMVisit1":
            database[sheets[0]].loc[database[sheets[0]]['Subject'] == row["pseudonym"], 'LEDD'] = 0
            database[sheets[1]].loc[database[sheets[1]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
        elif row["Timepoint"] == "ses-POMVisit2":
            database[sheets[2]].loc[database[sheets[2]]['Subject'] == row["pseudonym"], 'LEDD'] = 0
            database[sheets[3]].loc[database[sheets[3]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])
        elif row["Timepoint"] == "ses-POMVisit3":
            database[sheets[4]].loc[database[sheets[4]]['Subject'] == row["pseudonym"], 'LEDD'] = 0
            database[sheets[5]].loc[database[sheets[5]]['Subject'] == row["pseudonym"], 'LEDD'] = safe_convert_to_float(row["LEDD"])

    with pd.ExcelWriter(path_to_ppp_data_comp, engine='openpyxl') as writer:
        for sheet_name, data in database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    return database


def create_accelerometry_columns(database, sheets, path_to_ppp_data_comp):
    mat_data = pd.read_csv('/project/3024023.01/PPP-POM_cohort/Accelerometry/freq_power_scores.csv')
    for index, row in mat_data.iterrows():
        database[sheets[0]].loc[database[sheets[0]]['Subject'] == row["Sub"], 'FreqPeak'] = safe_convert_to_float(row["Freq"])
        database[sheets[1]].loc[database[sheets[1]]['Subject'] == row["Sub"], 'FreqPeak'] = safe_convert_to_float(row["Freq"])
        database[sheets[2]].loc[database[sheets[2]]['Subject'] == row["Sub"], 'FreqPeak'] = safe_convert_to_float(row["Freq"])
        database[sheets[3]].loc[database[sheets[3]]['Subject'] == row["Sub"], 'FreqPeak'] = safe_convert_to_float(row["Freq"])
        database[sheets[4]].loc[database[sheets[4]]['Subject'] == row["Sub"], 'FreqPeak'] = safe_convert_to_float(row["Freq"])
        database[sheets[5]].loc[database[sheets[5]]['Subject'] == row["Sub"], 'FreqPeak'] = safe_convert_to_float(row["Freq"])

        database[sheets[0]].loc[database[sheets[0]]['Subject'] == row["Sub"], 'LogPower'] = safe_convert_to_float(row["LogPower"])
        database[sheets[1]].loc[database[sheets[1]]['Subject'] == row["Sub"], 'LogPower'] = safe_convert_to_float(row["LogPower"])
        database[sheets[2]].loc[database[sheets[2]]['Subject'] == row["Sub"], 'LogPower'] = safe_convert_to_float(row["LogPower"])
        database[sheets[3]].loc[database[sheets[3]]['Subject'] == row["Sub"], 'LogPower'] = safe_convert_to_float(row["LogPower"])
        database[sheets[4]].loc[database[sheets[4]]['Subject'] == row["Sub"], 'LogPower'] = safe_convert_to_float(row["LogPower"])
        database[sheets[5]].loc[database[sheets[5]]['Subject'] == row["Sub"], 'LogPower'] = safe_convert_to_float(row["LogPower"])

    with pd.ExcelWriter(path_to_ppp_data_comp, engine='openpyxl') as writer:
        for sheet_name, data in database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    return database


if __name__ == "__main__":
    run_local = 1

    # Define base directories paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    # path_to_participants_list = os.path.join(base_PPP_folder, "PPP_cohort_participants-with-tremor-list.xlsx")
    path_to_participants_list = os.path.join(base_PPP_folder, "Complete_ppp_participants_list.xlsx")
    path_to_updrs_keys = os.path.join(base_PPP_folder, "updrs_keys.xlsx")
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

    freq_power = False
    if freq_power == False:
        ppp_database = create_accelerometry_columns(ppp_database, sheets, path_to_ppp_data_comp)

    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][0:33], "TotalU3")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][0:33], "AvgTotalU3", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][23:33], "TremorUPDRS")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][23:33], "AvgTremorUPDRS", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][7:17], "Brady5Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][7:17], "AvgBrady5Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][np.concatenate([np.arange(7, 20), [22]])], "Brady14Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][np.concatenate([np.arange(7, 20), [22]])], "AvgBrady14Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, [updrs_keys["FinalColumnsNames"][i] for i in [27, 28, 29, 30, 32]], "LimbsRestTrem")
    ppp_database = sum_keys(ppp_database, [updrs_keys["FinalColumnsNames"][i] for i in [27, 28, 29, 30, 32]], "AvgLimbsRestTrem", averageFlag=True)
    ppp_database = sum_keys(ppp_database, [updrs_keys["FinalColumnsNames"][i] for i in [27, 28, 29, 30]], "RestTrem")
    ppp_database = sum_keys(ppp_database, [updrs_keys["FinalColumnsNames"][i] for i in [27, 28, 29, 30]], "AvgRestTrem", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][3:7], "LimbsRigidity4Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][3:7], "AvgLimbsRigidity4Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][2:7], "LimbsRigidity5Items")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][2:7], "AvgLimbsRigidity5Items", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][23:25], "PosturalTremor")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][23:25], "AvgPosturalTremor", averageFlag=True)
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][25:27], "KineticTremor")
    ppp_database = sum_keys(ppp_database, updrs_keys["FinalColumnsNames"][25:27], "AvgKineticTremor", averageFlag=True)
    ppp_database = sum_keys(ppp_database, ["AvgBrady5Items", "AvgLimbsRigidity5Items"], "AvgLimbsBradyRig", averageFlag=True)
    ppp_database = sum_keys(ppp_database, ["Brady5Items", "LimbsRigidity5Items"], "LimbsBradyRig")

    ppp_database = dopamine_responsiveness(ppp_database,
                                           ["AvgTotalU3", "AvgTremorUPDRS", "AvgLimbsRestTrem", "AvgBrady14Items", "AvgLimbsRigidity5Items", "AvgLimbsBradyRig"],
                                           ["ChangeTotalU3", "ChangeTremorUPDRS", "ChangeLimbsRestTrem", "ChangeBrady14Items", "ChangeLimbsRigidity5Items", "ChangeLimbsBradyRig"],
                                           sheets, percentage_change_function=percentage_change_Basic)

    ppp_database = compute_tremor_responsiveness(ppp_database, ["RestTrem"], ["ResponseRestTrem"], sheets)

    """
    The metrics that I defined based on Zach's paper are:
    - Brady14Items
    - AvgBrady14Items
    - LimbsRestTrem
    - AvgLimbsRestTrem
    - LimbsRigidity5Items
    - AvgLimbsRigidity5Items
    
    The RestTrem column are the u17 elements, without constancy (u18).
    """

    with pd.ExcelWriter(path_to_ppp_data_comp, engine='openpyxl') as writer:
        for sheet_name, data in ppp_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print(f"----- DATASET WITH ADDITIONAL METRICS SAVED TO {path_to_ppp_data_comp} -----")
    sys.exit()
