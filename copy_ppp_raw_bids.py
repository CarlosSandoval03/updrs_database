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
    path_to_participants_list = os.path.join(base_PPP_folder, "bids", "subjects_handedness.xlsx")
    path_to_participants_list_tremor = os.path.join(base_PPP_folder, "PPP_cohort_participants-with-tremor-list.xlsx")

    if run_local: print("----- LOADING LIST OF PARTICIPANTS WITH TREMOR DURING SCANS -----")
    participants = extract_participants_list(path_to_participants_list)
    participants_tremor = extract_participants_list(path_to_participants_list_tremor)

    # participants_to_include = participants
    participants_to_include = pd.Series(list(set(participants_tremor) | set(participants)))

    session_names = ["ses-POMVisit1", "ses-POMVisit2", "ses-POMVisit3"]
    sub_folders = ["anat", "func"]
    name_pattern = ["_acq-MPRAGE_run-1_T1w", "_task-rest_acq-MB8_run-1_echo-1_bold"]

    searchObj = SearchDirectories()

    # For creating a summary of existance of data
    summary_sessions = pd.DataFrame()
    summary_sessions["Subject"] = participants_to_include.reset_index(drop=True)
    summary_sessions[session_names[0]] = pd.Series([0]*len(participants_to_include))
    summary_sessions[session_names[1]] = pd.Series([0] * len(participants_to_include))
    summary_sessions[session_names[2]] = pd.Series([0] * len(participants_to_include))

    for participant in participants_to_include:
        print(f"Subject processed: {participant}")
        
        for session in session_names:
            for i,folder in enumerate(sub_folders):
                full_path = os.path.join(path_to_ppp_bids, participant, session, folder)
                pattern = name_pattern[i]

                if os.path.exists(full_path):
                    summary_sessions.loc[summary_sessions['Subject'] == participant, session] = 1

                    files_dirs = searchObj.search_name_pattern(full_path, pattern)

                    # assert len(files_dirs) != 2; f"{len(files_dirs)} files found for {participant}-{session}-{folder}. Only 2 should be found."
                    if len(files_dirs) != 2:
                        print(f"{len(files_dirs)} files found for {participant}-{session}-{folder}. Only 2 should be found.")
                    else:
                        root_path_new = os.path.join(path_to_my_bids, participant, session, folder)
                        for orig_file in files_dirs:
                            _, name_of_file = os.path.split(orig_file)
                            out_dir = os.path.join(root_path_new, name_of_file)

                            if not os.path.exists(root_path_new):
                                os.makedirs(root_path_new)

                            shutil.copy(orig_file, out_dir)

    summary_sessions.to_excel(os.path.join(path_to_my_bids, "existance_of_sessions.xlsx"), index=False)

    sys.exit()