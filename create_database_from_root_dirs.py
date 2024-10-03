"""
This script access the root files from the PPP (POM) cohort and creates a CVS file
that includes all the UPDRS measurements for section 3 of the UPDRS-Analysis,
for every of the 139 participants that were identified with clear tremor.
"""

import os
import sys
import copy

import json
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime
import math


def extract_participants_list(file_path):
    df = pd.read_excel(file_path)
    participants = df["Subject"]

    return participants


def extract_updrs_dictionary_keys(path_to_updrs_keys):
    df = pd.read_excel(path_to_updrs_keys)
    columns_to_keep = ["FinalColumnsNames", "Description", "PPP_OFF_key", "PPP_ON_key", "PPP_JSON_number", "PPP_handness_relative_fields"]
    updrs_keys = df[columns_to_keep][0:34]

    return updrs_keys


def create_database_template(file_path, columns, books):
    book_df = []
    for i, _ in enumerate(books):
        book_df.append(pd.DataFrame(columns=columns))

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for i, book in enumerate(books):
            book_df[i].to_excel(writer, sheet_name=book, index=False)


def findCorrectJSONFile(path_to_sub_dirs, sub, visit_folder, json_off, json_on):
    path_to_off_json = ""
    path_to_on_json = ""

    for jsn in json_off:
        aux = os.path.join(path_to_sub_dirs, sub, visit_folder, jsn)
        if os.path.exists(aux):
            path_to_off_json = aux
            break
    for jsn in json_on:
        aux = os.path.join(path_to_sub_dirs, sub, visit_folder, jsn)
        if os.path.exists(aux):
            path_to_on_json = aux
            break

    return path_to_off_json, path_to_on_json


def fill_entry_to_database(new_entry, sub, path_to_json, jsonIdx, indexesKeys, OfOn):
    new_entry["Subject"] = sub
    if OfOn == "Of":
        sesFlag = "OFF"
    else:
        sesFlag = "ON"

    with open(path_to_json, 'r') as file:
        dataJSON = json.load(file)
    if jsonIdx == 1:
        handDevice = dataJSON["crf"].get(f"Up3{OfOn}DeviSide", np.nan)
        if ("linker" in handDevice) or ("2" in handDevice):
            new_entry["DeviceHand"] = 2
        elif ("rechter" in handDevice) or ("1" in handDevice):
            new_entry["DeviceHand"] = 1
        else:
            new_entry["DeviceHand"] = np.nan


    for idx in indexesKeys:
        key = updrs_keys["FinalColumnsNames"][idx]
        if updrs_keys[f"PPP_{sesFlag}_key"][idx] == "Device_allocation_dependant":
            fieldName = updrs_keys["PPP_handness_relative_fields"][idx]
            fieldName = fieldName.replace("%", OfOn)
            if (new_entry["DeviceHand"] == 2) & ("Left" in updrs_keys["Description"][idx]):  # Left-Left
                fieldName = fieldName.replace("&", "YesDev")
            elif (new_entry["DeviceHand"] == 2) & ("Right" in updrs_keys["Description"][idx]):  # Left-Right
                fieldName = fieldName.replace("&", "NonDev")
            elif (new_entry["DeviceHand"] == 1) & ("Left" in updrs_keys["Description"][idx]):  # Right-Left
                fieldName = fieldName.replace("&", "NonDev")
            elif (new_entry["DeviceHand"] == 1) & ("Right" in updrs_keys["Description"][idx]):  # Right-Right
                fieldName = fieldName.replace("&", "YesDev")
            value = dataJSON["crf"].get(fieldName, np.nan)
        else:
            value = dataJSON["crf"].get(updrs_keys[f"PPP_{sesFlag}_key"][idx], np.nan)

        try:
            value = int(value)
        except ValueError:
            value = np.nan

        new_entry[key] = value

    return new_entry


def safe_convert_to_int(x):
    try:
        return int(x)
    except ValueError:
        return np.nan


def load_if_exists(path, keys, dict_data, outKeyO = [""]):
    if os.path.exists(path):
        with open(path, 'r') as file:
            dataJSON = json.load(file)
        for i, key in enumerate(keys):
            if outKeyO[0] == "":
                outKey = key
            else:
                outKey = outKeyO[i]
            dict_data[outKey] = safe_convert_to_int(dataJSON["crf"].get(key, np.nan))
    else:
        for i, key in enumerate(keys):
            if outKeyO[0] == "":
                outKey = key
            else:
                outKey = outKeyO[i]
            dict_data[outKey] = np.nan

    return dict_data


def parse_time_string(time_str):
    if isinstance(time_str, float) and math.isnan(time_str):
        return np.nan
    if "USER_MISSING" in time_str:
        return np.nan
    if ';' in time_str:
        # Format with date
        time_part = time_str.split(';')[1]
    else:
        # Format without date
        time_part = time_str
    return datetime.strptime(time_part, "%H:%M")


def include_diagnose_information_and_manual_keys(dict_data, path_to_sub_dirs, sub, visit_folder, visit_idx, session=["offon", ""]):
    if session[0] == "offon":
        algemeen1_name = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx+1}.Motorische_taken_OFF.Algemeen1.json")
        algemeen1_keys = ["FirstSympYear", "FirstSympMonth", "PrefHand", "MostAffSide"]
        dict_data = load_if_exists(algemeen1_name, algemeen1_keys, dict_data, outKeyO=["FirstSympYear", "FirstSympMonth", "Handedness", "MostAffSide"])

        algemeen3_name = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx+1}.Motorische_taken_OFF.Algemeen3.json")
        algemeen3_keys = ["DiagParkMonth"]
        dict_data = load_if_exists(algemeen3_name, algemeen3_keys, dict_data)

        algemeen4_name = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx+1}.Motorische_taken_OFF.Algemeen4.json")
        algemeen4_keys = ["DiagParkYear"]
        dict_data = load_if_exists(algemeen4_name, algemeen4_keys, dict_data)

        demo_quest_name = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx+1}.Demografische_vragenlijsten.Part1.json")
        demo_quest_keys = ["Age", "Gender"]
        dict_data = load_if_exists(demo_quest_name, demo_quest_keys, dict_data)

        base_res_name = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx+1}.Demografische_vragenlijsten.Basaal_onderzoek.json")
        base_res_keys = ["BodMasInd", "Length", "Weight"]
        dict_data = load_if_exists(base_res_name, base_res_keys, dict_data)
    elif session[0] == "off":
        handy = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx + 1}.Motorische_taken_OFF.Hoehn__Yahr_stage.json")
        handy_keys = ["Up3OfHoeYah"]
        dict_data = load_if_exists(handy, handy_keys, dict_data, outKeyO=["HoeYah"])
        if os.path.exists(session[1]):
            with open(session[1], 'r') as file:
                dataJSON = json.load(file)
            dict_data["DoseLagMinutes"] = dataJSON["crf"].get("Up3OfDoseLagMinutes", np.nan)
            dict_data["AssessYear"] = dataJSON["crf"].get("AssessYear", np.nan)
            dict_data["MonthSinceDiag"] = dataJSON["crf"].get("MonthSinceDiag", np.nan)
        else:
            dict_data["DoseLagMinutes"] = np.nan
            dict_data["AssessYear"] = np.nan
            dict_data["MonthSinceDiag"] = np.nan
    elif session[0] == "on":
        handy = os.path.join(path_to_sub_dirs, sub, visit_folder, f"Castor.Visit{visit_idx + 1}.Motorische_taken_ON.Hoehn__Yahr_stage.json")
        handy_keys = ["Up3OnHoeYah"]
        dict_data = load_if_exists(handy, handy_keys, dict_data, outKeyO=["HoeYah"])
        if os.path.exists(session[1]):
            with open(session[1], 'r') as file:
                dataJSON = json.load(file)
            time1_obj = parse_time_string(dataJSON["crf"].get("Up3OnMedicTime", np.nan))
            time2_obj = parse_time_string(dataJSON["crf"].get("Up3OnAssesTime", np.nan))
            if pd.isna(time1_obj) or pd.isna(time2_obj):
                minutes_difference = np.nan
            else:
                time_difference = time2_obj - time1_obj
                minutes_difference = time_difference.total_seconds() / 60
            dict_data["DoseLagMinutes"] = minutes_difference
            dict_data["AssessYear"] = np.nan
            dict_data["MonthSinceDiag"] = np.nan
        else:
            dict_data["DoseLagMinutes"] = np.nan
            dict_data["AssessYear"] = np.nan
            dict_data["MonthSinceDiag"] = np.nan

    return dict_data


def fill_updrs_scores(path_to_updrs_database, updrs_keys, participants, path_to_sub_dirs):
    excel_file = pd.ExcelFile(path_to_updrs_database)
    sheets = excel_file.sheet_names
    df = {}
    for sheet in sheets:
        df[sheet] = pd.read_excel(path_to_updrs_database, sheet_name=sheet)

    visits_folders = ["ses-POMVisit1", "ses-POMVisit2", "ses-POMVisit3"]
    visits = [1, 2, 3]
    jsonIdxes = [1, 2, 3]
    OnOffVar = ["OFF", "ON"]

    for sub in participants:
        if run_local: print(f"----- FILLING DATA FOR {sub} -----")
        for i, visit_folder in enumerate(visits_folders):
            new_entry_off = {}
            new_entry_on = {}
            for jsonIdx in jsonIdxes:
                json_off = [
                    f"Castor.Visit{visits[i]}.Motorische_taken_{OnOffVar[0]}.Updrs3_deel_{jsonIdx}.json",
                    f"Castor.Visit{visits[i]}.Motorische_taken_{OnOffVar[0]}.Updrs_3_deel_{jsonIdx}.json"
                    ]
                json_on = [
                    f"Castor.Visit{visits[i]}.Motorische_taken_{OnOffVar[1]}.Updrs3_deel_{jsonIdx}.json",
                    f"Castor.Visit{visits[i]}.Motorische_taken_{OnOffVar[1]}.Updrs_3_deel_{jsonIdx}.json"
                    ]

                [path_to_off_json, path_to_on_json] = findCorrectJSONFile(path_to_sub_dirs, sub, visit_folder, json_off, json_on)
                if jsonIdx == 1:
                    path_off_manual = copy.deepcopy(path_to_off_json)
                    path_on_manual = copy.deepcopy(path_to_on_json)

                indexesKeys = updrs_keys[updrs_keys["PPP_JSON_number"] == jsonIdx].index.tolist()

                if os.path.exists(path_to_off_json):
                    new_entry_off = fill_entry_to_database(new_entry_off, sub, path_to_off_json, jsonIdx, indexesKeys, OfOn="Of")
                else:
                    new_entry_off["Subject"] = sub
                    new_entry_off["DeviceHand"] = np.nan
                    for column_name in updrs_keys["FinalColumnsNames"]:
                        new_entry_off[column_name] = np.nan

                if os.path.exists(path_to_on_json):
                    new_entry_on = fill_entry_to_database(new_entry_on, sub, path_to_on_json, jsonIdx, indexesKeys, OfOn="On")
                else:
                    new_entry_on["Subject"] = sub
                    new_entry_on["DeviceHand"] = np.nan
                    for column_name in updrs_keys["FinalColumnsNames"]:
                        new_entry_on[column_name] = np.nan

            # Here include diagnose information and manual keys
            data_diagnose = {}
            data_diagnose = include_diagnose_information_and_manual_keys(data_diagnose, path_to_sub_dirs, sub, visit_folder, i, session=["offon", ""])
            for key, value in data_diagnose.items():
                new_entry_off[key] = value
                new_entry_on[key] = value

            data_diagnose = {}
            data_diagnose = include_diagnose_information_and_manual_keys(data_diagnose, path_to_sub_dirs, sub, visit_folder, i, session=["off", path_off_manual])
            for key, value in data_diagnose.items():
                new_entry_off[key] = value

            data_diagnose = {}
            data_diagnose = include_diagnose_information_and_manual_keys(data_diagnose, path_to_sub_dirs, sub, visit_folder, i, session=["on", path_on_manual])
            for key, value in data_diagnose.items():
                new_entry_on[key] = value
            # End of diagnose information and manual keys

            new_entry_off = pd.DataFrame([new_entry_off])
            new_entry_on = pd.DataFrame([new_entry_on])
            if i == 0:
                df[sheets[0]] = pd.concat([df[sheets[0]], new_entry_off], ignore_index=True)
                df[sheets[1]] = pd.concat([df[sheets[1]], new_entry_on], ignore_index=True)
            elif i == 1:
                df[sheets[2]] = pd.concat([df[sheets[2]], new_entry_off], ignore_index=True)
                df[sheets[3]] = pd.concat([df[sheets[3]], new_entry_on], ignore_index=True)
            elif i == 2:
                df[sheets[4]] = pd.concat([df[sheets[4]], new_entry_off], ignore_index=True)
                df[sheets[5]] = pd.concat([df[sheets[5]], new_entry_on], ignore_index=True)

            with pd.ExcelWriter(path_to_updrs_database, engine='openpyxl') as writer:
                for sheet_name, data in df.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    run_local = 1

    # Define base directories paths
    base_PPP_folder             = "/project/3024023.01/PPP-POM_cohort/"
    # path_to_participants_list   = os.path.join(base_PPP_folder, "PPP_cohort_participants-with-tremor-list.xlsx")
    path_to_participants_list = os.path.join(base_PPP_folder, "Complete_ppp_participants_list.xlsx")
    path_to_updrs_keys          = os.path.join(base_PPP_folder, "updrs_keys.xlsx")
    path_to_updrs_database      = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_raw_data.xlsx")
    path_to_sub_dirs            = "/project/3022026.01/pep/ClinVars_10-08-2023/"

    if run_local: print("----- LOADING PARTICIPANTS LIST -----")
    participants = extract_participants_list(path_to_participants_list)

    if run_local: print("----- LOADING UPDRS DICTIONARIES KEYS -----")
    updrs_keys = extract_updrs_dictionary_keys(path_to_updrs_keys)

    if run_local: print("----- CREATING DATABASE TEMPLATE -----")
    books_database = ["OFF session - Visit 1", "ON session - Visit 1", "OFF session - Visit 2", "ON session - Visit 2", "OFF session - Visit 3", "ON session - Visit 3"]
    database_columns = updrs_keys["FinalColumnsNames"].tolist()
    database_columns.insert(0, "Subject")
    database_columns.insert(1, "DeviceHand")
    create_database_template(path_to_updrs_database, database_columns, books_database)

    if run_local: print("----- FILLING DATABASE -----")
    fill_updrs_scores(path_to_updrs_database, updrs_keys, participants, path_to_sub_dirs)

    if run_local: print(f"----- DATABASE CREATED TO {path_to_updrs_database} -----")

    sys.exit()
