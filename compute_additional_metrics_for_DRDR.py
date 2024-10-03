import os
import sys
import copy
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats
from scipy.stats import t
from scipy.special import gammaln
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from scipy.io import loadmat


def percentage_change_Elble(ratingOn, ratingOff, alpha=0.5):
    if len(ratingOn) != len(ratingOff):
        raise ValueError("Rating ON and Rating OFF have different number of elements.")
    if len(ratingOn) > 1:
        change = [1*(pow(10, alpha*(ratingOn[i] - ratingOff[i]))-1) for i, _ in enumerate(ratingOn)]
    else:
        change = 1*(pow(10, alpha*(ratingOn - ratingOff))-1)
    return [-1 * x for x in change]


def percentage_change_Basic(ratingOn, ratingOff, alpha=0.5):
    if len(ratingOn) != len(ratingOff):
        raise ValueError("Rating ON and Rating OFF have different number of elements.")
    if len(ratingOn) > 1:
        change = [
            1 * ((ratingOff[i] - ratingOn[i]) / ratingOff[i]) if ratingOff[i] != 0 else 0
            for i in range(len(ratingOn))
        ]
    else:
        change = 1*((ratingOff - ratingOn) / ratingOff)
    return change #[-1 * x for x in change]


def dopamine_responsiveness(database, keys, column_name, sheets, percentage_change_function=percentage_change_Elble):
    # Computes the OFF - ON difference per visit.
    valueON = None
    valueOFF = None
    for i, key in enumerate(keys):
        idx = 0
        for sheet_name, data in database.items():
            if "OFF" in sheet_name:
                filteredData = data[key["off"]]
                valueOFF = filteredData
            elif "ON" in sheet_name:
                filteredData = data[key["on"]]
                valueON = filteredData

            if valueON is not None and valueOFF is not None:
                responsiveness = percentage_change_function(valueON, valueOFF)
                database[sheets[idx]][column_name[i]] = responsiveness
                database[sheets[idx-1]][column_name[i]] = responsiveness
                valueON = None
                valueOFF = None

            idx=idx+1
    return database


def compute_tremor_responsiveness(database, keys, column_name, sheets):
    # Computes the OFF - ON difference per visit.
    visits = [0]
    for i, key in enumerate(keys):
        for visit in visits:
            filteredData_off = database[sheets[2*visit]][key]
            weightsData_off = database[sheets[2 * visit]]["OFFU18"] / 4
            filteredData_on = database[sheets[2 * visit + 1]][key]
            weightsData_on = database[sheets[2 * visit + 1]]["ONU18"] / 4

            weightsData_off = weightsData_off.replace(0, 0.25)
            weightsData_on = weightsData_on.replace(0, 0.25)

            responsiveness = (filteredData_off * weightsData_off) - (filteredData_on * weightsData_on)

            database[sheets[2 * visit]]["WRestTrem"] = filteredData_off * weightsData_off
            database[sheets[2 * visit + 1]]["WRestTrem"] = filteredData_on * weightsData_on
            database[sheets[2 * visit]][column_name[i]] = responsiveness
            database[sheets[2 * visit + 1]][column_name[i]] = responsiveness

    return database


def safe_convert_to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def create_power_columns(database, sheets, path_to_ppp_data_comp):
    powers = pd.read_csv("/project/3024023.01/fMRI_DRDR/EMG/drdr_tremor_power_compilation.csv")
    for index, row in powers.iterrows():
        if row.Session == 1:
            database[sheets[0]].loc[database[sheets[0]]['PatCode'] == f"DRDR_PD{row['Subject']}", f"power_{row['Condition']}"] = safe_convert_to_float(row["Power"])
        else:
            database[sheets[1]].loc[database[sheets[1]]['PatCode'] == f"DRDR_PD{row['Subject']}", f"power_{row['Condition']}"] = safe_convert_to_float(row["Power"])

    # with pd.ExcelWriter(path_to_ppp_data_comp, engine='openpyxl') as writer:
    #     for sheet_name, data in database.items():
    #         data.to_excel(writer, sheet_name=sheet_name, index=False)
    return database


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


if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_drdr_folder = "/project/3024023.01/fMRI_DRDR/"
    results_folder_drdr = "/project/3024023.01/fMRI_DRDR/updrs_analysis/"
    path_to_drdr_data = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean_complete.xlsx")
    path_to_drdr_data_comp = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean_complete.xlsx")

    if run_local: print("----- LOADING DATABASE DRDR -----")
    excel_file_drdr = pd.ExcelFile(path_to_drdr_data)
    sheets_drdr = excel_file_drdr.sheet_names
    drdr_database = {}
    for sheet in sheets_drdr:
        if sheet in ["UPDRS OFF", "UPDRS ON"]:
            drdr_database[sheet] = pd.read_excel(path_to_drdr_data, sheet_name=sheet)
    sheets_drdr = ["UPDRS OFF", "UPDRS ON"]

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    keys_data = [
        {"off": "AvgTotalU3", "on": "AvgTotalU3"},
        {"off": "AvgTremorUPDRS", "on": "AvgTremorUPDRS"},
        {"off": "AvgLimbsRestTrem", "on": "AvgLimbsRestTrem"},
        {"off": "AvgBrady14Items", "on": "AvgBrady14Items"},
        {"off": "AvgLimbsRigidity5Items", "on": "AvgLimbsRigidity5Items"}
    ]
    # name_columns = ["ResponseTotalU3", "ResponseTremorUPDRS", "ResponseLimbsRestTrem", "ResponseBrady14Items", "ResponseLimbsRigidity5Items"]
    # drdr_database = dopamine_responsiveness(drdr_database, keys_data, name_columns, sheets_drdr, percentage_change_function=percentage_change_Elble)

    drdr_database = compute_tremor_responsiveness(drdr_database, ["RestTrem"], ["ResponseRestTrem"], sheets_drdr)

    # drdr_database = create_power_columns(drdr_database, sheets_drdr, path_to_drdr_data_comp)

    with pd.ExcelWriter(path_to_drdr_data_comp, engine='openpyxl') as writer:
        for sheet_name, data in drdr_database.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    if run_local: print(f"----- DATASET WITH ADDITIONAL METRICS SAVED TO {path_to_drdr_data_comp} -----")