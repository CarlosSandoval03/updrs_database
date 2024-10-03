import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg'

"""
This code extracts the response predictors from every sheet in the database and puts them in a 
single sheet, to be used in SPSS.
"""

if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(results_folder_ppp, "ppp_updrs_database_consistent_subjects_trem-filtered.xlsx")
    # path_to_ppp_data_reduced = os.path.join(results_folder_ppp, "ppp_updrs_database_consistent_subjects.xlsx")

    path_to_output_database = os.path.join(results_folder_ppp, "ppp_updrs_trem-filtered_database_of_predictors.xlsx")

    if run_local: print("----- LOADING DATABASE PPP -----")
    excel_file_ppp = pd.ExcelFile(path_to_ppp_data_reduced)
    sheets_ppp = excel_file_ppp.sheet_names
    ppp_database = {}
    for sheet in sheets_ppp:
        ppp_database[sheet] = pd.read_excel(path_to_ppp_data_reduced, sheet_name=sheet)

    predictors_of_interest = ["ResponseRestTrem", "ChangeTremorUPDRS", "ChangeLimbsRestTrem", "ChangeTotalU3", "ChangeBrady14Items", "ChangeLimbsRigidity5Items", "ChangeLimbsBradyRig", "LogPower"]
    sheets_of_interest = [1, 2, 3]

    database_of_predictors = pd.DataFrame()
    subjects_ids = ppp_database[sheets_ppp[0]]["Subject"]
    database_of_predictors = pd.concat([database_of_predictors, subjects_ids], axis=1)

    for sh in sheets_of_interest:
        df = ppp_database[sheets_ppp[2*sh-1]][predictors_of_interest]
        df = df.rename(columns={col: f"{col}_{sh}" for col in df.columns})
        database_of_predictors = pd.concat([database_of_predictors, df], axis=1)


    database_of_predictors.to_excel(path_to_output_database, index=False)


