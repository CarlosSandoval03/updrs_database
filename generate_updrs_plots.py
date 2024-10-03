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
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg'
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Agg'

import matplotlib.collections as clt

# My Library: Codebase
sys.path.append(os.path.expanduser('~/PythonProjects/Codebase'))
from utilitiespkg import DataHandling
from clusteringpkg import Clustering
from plottingpkg import UPDRSPlotting

filterHandsTremorSubjects = DataHandling.get_subjects_above_threshold
percentage_change_Elble = DataHandling.percentage_change_elble
percentage_change_Basic = DataHandling.percentage_change_basic
cleanNaNRowsifAnyMult = DataHandling.remove_rows_with_nans


def pdf_plots(database, conf_dicts, sheets, results_folder, analysis_name, style="all", dataset="ppp", filterHandsTremorFlag=True):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    df = copy.deepcopy(database)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)
    for confs in conf_dicts:
        typeC = confs['type']
        for i_data, dataX in enumerate(confs['x']):
            if typeC == "offon":
                for visit in confs["visit"]:
                    fig = go.Figure()
                    plt.figure(figsize=(10, 6))
                    colIdx = 0
                    if filterHandsTremorFlag == True:
                        idx = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], 1, df[sheets[2*(visit-1)]])
                        offD = pd.Series([df[sheets[2*(visit-1)]][dataX['off']][i] for i in idx])
                        onD = pd.Series([df[sheets[2*(visit-1)+1]][dataX['on']][i] for i in idx])
                    else:
                        offD = df[sheets[2*(visit-1)+1]][dataX['off']]
                        onD = df[sheets[2*(visit-1)+1]][dataX['on']]

                    cleaned_series = cleanNaNRowsifAnyMult(offD, onD)
                    offD, onD = cleaned_series[0], cleaned_series[1]

                    if style == "all" or style == "plotly":
                        # For histogram like plots
                        # fig.add_trace(go.Histogram(x=offD, histnorm='probability density', name='OFF', opacity=0.5, nbinsx=30))
                        # fig.add_trace(go.Histogram(x=onD, histnorm='probability density', name='ON', opacity=0.5, nbinsx=30))

                        # For Smoothed lines with shadow
                        fig.add_trace(create_kde(offD, 'rgba(255, 0, 0, 0.3)', 'OFF'))
                        fig.add_trace(create_kde(onD, 'rgba(0, 0, 255, 0.3)', 'ON'))

                        # For just lines
                        # fig = ff.create_distplot([offD, onD], ["OFF", "ON"], show_hist=False, colors=['blue', 'red'], show_rug=False)

                        fig.update_layout(title='Probability Density Function (PDF)',
                                          xaxis_title=f"{dataX['off']}",
                                          yaxis_title='Density',
                                          barmode='overlay')
                        figure_name = os.path.join(results_folder, analysis_name, f"pdf_{dataX['off']}_OffOn_plotly.png")
                        fig.write_image(figure_name)
                        del fig
                    if style == "all" or style == "sns":
                        sns.kdeplot(data=offD, label='OFF', color='blue', alpha=0.3, bw_adjust=0.5, fill=True)
                        sns.kdeplot(data=onD, label='ON', color='red', alpha=0.3, bw_adjust=0.5, fill=True)
                        plt.title('Probability Density Function (PDF)')
                        plt.xlabel(f"{dataX['off']}")
                        plt.ylabel('Density')
                        plt.legend(title='Sessions')
                        figure_name = os.path.join(results_folder, analysis_name, f"pdf_{dataX['off']}_OffOn_sns.png")
                        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                        plt.clf()
                        plt.close()
            elif typeC == "longitudinal":
                for ses in confs["ses"]:
                    fig = go.Figure()
                    plt.figure(figsize=(10, 6))
                    colIdx = 0
                    if filterHandsTremorFlag == True:
                        idx = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], 1, df[sheets[0]], df[sheets[2]], df[sheets[4]])
                        d1 = pd.Series([df[sheets[0]][dataX[ses]][i] for i in idx])
                        d2 = pd.Series([df[sheets[2]][dataX[ses]][i] for i in idx])
                        d3 = pd.Series([df[sheets[4]][dataX[ses]][i] for i in idx])
                    else:
                        d1 = df[sheets[0]][dataX[ses]]
                        d2 = df[sheets[2]][dataX[ses]]
                        d3 = df[sheets[4]][dataX[ses]]

                    cleaned_series = cleanNaNRowsifAnyMult(d1, d2, d3)
                    d1, d2, d3 = cleaned_series[0], cleaned_series[1], cleaned_series[2]

                    if style == "all" or style == "plotly":
                        # For histogram like plots
                        # fig.add_trace(go.Histogram(x=d1, histnorm='probability density', name='Baseline', opacity=0.5, nbinsx=30))
                        # fig.add_trace(go.Histogram(x=d2, histnorm='probability density', name='Year 1', opacity=0.5, nbinsx=30))
                        # fig.add_trace(go.Histogram(x=d3, histnorm='probability density', name='Year 2', opacity=0.5, nbinsx=30))

                        # For smoothed lines with shadow
                        fig.add_trace(create_kde(d1, 'rgba(0, 0, 255, 0.3)', 'Baseline'))  # blue with opacity
                        fig.add_trace(create_kde(d2, 'rgba(255, 0, 0, 0.3)', 'Year 1'))  # red with opacity
                        fig.add_trace(create_kde(d3, 'rgba(0, 255, 0, 0.3)', 'Year 2'))  # green with opacity

                        # For lines only
                        # fig = ff.create_distplot([d1, d2, d3], ["Baseline", "Year 1", "Year 2"], show_hist=False, colors=['blue', 'red', 'green'], show_rug=False)

                        fig.update_layout(title='Probability Density Function (PDF)',
                                          xaxis_title=f"{dataX['off']}",
                                          yaxis_title='Density',
                                          barmode='overlay')
                        figure_name = os.path.join(results_folder, analysis_name, f"pdf_{dataX['off']}_long_plotly.png")
                        fig.write_image(figure_name)
                        del fig
                    if style == "all" or style == "sns":
                        sns.kdeplot(data=d1, label='Baseline', color='blue', alpha=0.3, bw_adjust=0.5, fill=True)
                        sns.kdeplot(data=d2, label='Year 1', color='red', alpha=0.3, bw_adjust=0.5, fill=True)
                        sns.kdeplot(data=d3, label='Year 2', color='green', alpha=0.3, bw_adjust=0.5, fill=True)
                        plt.title('Probability Density Function (PDF)')
                        plt.xlabel(f"{dataX['off']}")
                        plt.ylabel('Density')
                        plt.legend(title='Sessions')
                        figure_name = os.path.join(results_folder, analysis_name, f"pdf_{dataX['off']}_long_sns.png")
                        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
                        plt.clf()
                        plt.close()
            else:
                raise ValueError("Select an appropriate type of plotting PDF: OFF-ON or Longitudinal comparisons.")

    return 0


def create_kde(series, color, label):
    kde = sns.kdeplot(series, bw_adjust=0.5)
    x, y = kde.get_lines()[0].get_data()
    plt.close()
    return go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', fillcolor=color, line=dict(color=color, width=2), name=label)


def coeffs_diff_plot(metric_function, metric_name, data, updrs_conf, sheets, results_folder, analysis_name, dataset="ppp", filterHandsTremorFlag=True):
    database = copy.deepcopy(data)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)
    dx = "Visit"
    dhue = "Group"

    for confs in updrs_conf:
        plt.figure(figsize=(10, 6))
        df = {}
        df[dx] = pd.Series(dtype='str')
        df[dhue] = pd.Series(dtype='str')
        updrs_keys = confs["updrs"]
        if len(updrs_keys)>1:
            nameUPDRSkeys = ""
            for updrs_key in updrs_keys:
                df[updrs_key["off"]] = pd.Series(dtype='float')
                nameUPDRSkeys = nameUPDRSkeys + updrs_key["off"]
        else:
            df[updrs_keys[0]["off"]] = pd.Series(dtype='float')
            nameUPDRSkeys = updrs_keys[0]["off"]
        timeName, sesName = getNamesOfComparisons(confs["visit"], confs["ses"])

        for visit in confs["visit"]:
            for ses in confs["ses"]:
                sheet = 2 * (visit - 1) if ses == "off" else 2 * (visit - 1) + 1
                if filterHandsTremorFlag:
                    if dataset == "ppp":
                        idx = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], 1, database[sheets[2*(visit-1)]])
                    elif dataset == "drdr":
                        idx = filterHandsTremorSubjects(["OFFU17RUE", "OFFU17LUE"], 1, database[sheets[2 * (visit - 1)]])

                for updrs in updrs_keys:
                    key = updrs["off"]
                    data_series = database[sheets[sheet]][updrs[ses]]

                    if filterHandsTremorFlag:
                        data_series = pd.Series([data_series[i] for i in idx])

                    if not df[key].empty:
                        df[key] = pd.concat([df[key], data_series], ignore_index=True)
                    else:
                        df[key] = data_series

                lengthData = len(data_series)

                visit_label = {
                    1: "Baseline" if dataset == "ppp" else "DRDR",
                    2: "Year 1",
                    3: "Year 2"
                }.get(visit, "Unknown")
                df[dx] = pd.concat([df[dx], pd.Series([visit_label] * lengthData)], ignore_index=True)

                ses_label = "OFF" if ses == "off" else "ON"
                df[dhue] = pd.concat([df[dhue], pd.Series([ses_label] * lengthData)], ignore_index=True)

        if isinstance(data, dict):
            df = pd.DataFrame(df)

        cleaned = cleanNaNRowsifAnyMult(df[key], df[dx], df[dhue])
        df[key], df[dx], df[dhue] = cleaned[0], cleaned[1], cleaned[2]

        df_melted = df.melt(id_vars=[dx, dhue], var_name='UPDRS', value_name='value')
        g = sns.FacetGrid(df_melted, col='UPDRS', hue=dhue, sharey=False, height=5, aspect=1.5)
        g.map(sns.pointplot, dx, 'value', dodge=True, capsize=0.1, errorbar='sd', estimator=metric_function)
        g.add_legend(title=dhue)
        g.set_axis_labels(dx, "UPDRS Score")
        g.set_titles("{col_name}")

        # Perform ANOVA for each measurement
        for measurement in df_melted['UPDRS'].unique():
            formula = f'value ~ C({dx}) + C({dhue}) + C({dx}):C({dhue})'
            model = ols(formula, data=df_melted[df_melted['UPDRS'] == measurement]).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_table.to_excel(os.path.join(results_folder, analysis_name, f"ANOVA results for {measurement}_{timeName}-{sesName}.xlsx"), engine='openpyxl', index=True)

        if dataset == "ppp":
            figure_name = os.path.join(results_folder, analysis_name, f"cd_{metric_name}_{nameUPDRSkeys}_{timeName}-{sesName}_sns.png")
        else:
            figure_name = os.path.join(results_folder, analysis_name, f"cd_{nameUPDRSkeys}_{sesName}_sns.png")
        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
        plt.clf()
        plt.close()
    return 0


def single_sub_trend_plot(data, updrs_conf, sheets, results_folder, analysis_name, dataset="ppp", filterHandsTremorFlag=True):
    # Instead of group I have sub-ID
    database = copy.deepcopy(data)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)

    subIDKey = "Subject" if dataset == "ppp" else "PatCode"
    dx = "Visit"
    dhue = subIDKey

    for confs in updrs_conf:
        df = {}
        df[dx] = pd.Series(dtype='str')
        df[dhue] = pd.Series(dtype='str')
        meansTest = []
        updrs_keys = confs["updrs"]
        if len(updrs_keys)>1:
            nameUPDRSkeys = ""
            for updrs_key in updrs_keys:
                df[updrs_key["off"]] = pd.Series(dtype='float')
                nameUPDRSkeys = nameUPDRSkeys + updrs_key["off"]
        else:
            df[updrs_keys[0]["off"]] = pd.Series(dtype='float')
            nameUPDRSkeys = updrs_keys[0]["off"]
        timeName, sesName = getNamesOfComparisons(confs["visit"], confs["ses"])

        for visit in confs["visit"]:
            sheetOFF = 2 * (visit - 1) if dataset == "ppp" else 0
            sheetON = 2 * (visit - 1) + 1 if dataset == "ppp" else 1
            if filterHandsTremorFlag:
                idx = filterHandsTremorSubjects(["U17ResTremRUE", "U17ResTremLUE"], 1, database[sheets[sheetOFF]])

            for updrs in updrs_keys:
                key = updrs["off"]
                data_seriesOFF = database[sheets[sheetOFF]][updrs["off"]]
                data_seriesON = database[sheets[sheetON]][updrs["on"]]

                if filterHandsTremorFlag:
                    data_seriesOFF = pd.Series([data_seriesOFF[i] for i in idx])
                    data_seriesON = pd.Series([data_seriesON[i] for i in idx])

                data_series = data_seriesOFF - data_seriesON

                if not df[key].empty:
                    df[key] = pd.concat([df[key], data_series], ignore_index=True)
                else:
                    df[key] = data_series

                meansTest.append(np.mean(data_series))

            lengthData = len(data_series)

            visit_label = {
                1: "Baseline" if dataset == "ppp" else "DRDR",
                2: "Year 1",
                3: "Year 2"
            }.get(visit, "Unknown")
            df[dx] = pd.concat([df[dx], pd.Series([visit_label] * lengthData)], ignore_index=True)


            if filterHandsTremorFlag:
                subIDs = pd.Series([database[sheets[sheetOFF]][subIDKey][i] for i in idx])
            else:
                subIDs = database[sheets[sheetOFF]][subIDKey]
            df[dhue] = pd.concat([df[dhue], subIDs], ignore_index=True)

        if isinstance(data, dict):
            df = pd.DataFrame(df)

        cleaned = cleanNaNRowsifAnyMult(df[key], df[dx], df[dhue])
        df[key], df[dx], df[dhue] = cleaned[0], cleaned[1], cleaned[2]

        # To add Mean lines
        df_melted = df.melt(id_vars=[dx, dhue], var_name='UPDRS', value_name='value')
        g = sns.FacetGrid(df_melted, col='UPDRS', hue=dhue, sharey=False, height=5, aspect=1.5)
        dataMeans = df_melted.groupby([dx, 'UPDRS']).agg(mean_value=('value', 'mean')).reset_index()
        g.map_dataframe(plot_individual_lines, dx=dx, dhue=dhue, dataMeans=dataMeans, linewidth=1)
        g.set_axis_labels(dx, "UPDRS Score")
        g.set_titles("{col_name}")
        for ax in g.axes.flatten():
            ax.grid(True)


        if dataset == "ppp":
            figure_name = os.path.join(results_folder, analysis_name, f"cd_{nameUPDRSkeys}_{timeName}-{sesName}_sns.png")
        else:
            figure_name = os.path.join(results_folder, analysis_name, f"cd_{nameUPDRSkeys}_{sesName}_sns.png")
        plt.savefig(figure_name, bbox_inches='tight', dpi=600)
        plt.clf()
        plt.close()
    return 0


def plot_individual_lines(data, dx, dhue, dataMeans=None, linewidth=1.5, mean_line_color='black', mean_line_thickness=2, **kwargs):
    ax = plt.gca()
    # Plot individual lines
    participants = data[dhue].unique()
    for participant in participants:
        sns.lineplot(data=data[data[dhue] == participant], x=dx, y='value',
                     errorbar=None, marker='o', linewidth=linewidth, ax=ax, **kwargs)

    # Plot mean line if provided
    offset = (np.max(dataMeans["mean_value"])-np.min(dataMeans["mean_value"]))/5
    if dataMeans is not None:
        for updrs in data['UPDRS'].unique():
            mean_data = dataMeans[(dataMeans['UPDRS'] == updrs) & (dataMeans[dx].isin(data[dx].unique()))]
            ax.plot(mean_data[dx], mean_data['mean_value'], color=mean_line_color, linewidth=mean_line_thickness,
                    label='Mean', linestyle='--', marker='s', markersize=5)

            for _, row in mean_data.iterrows():
                ax.text(row[dx], row['mean_value']+offset, f"{row['mean_value']:.3f}", color='black',
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(dx)
    ax.set_ylabel('UPDRS Score')
    ax.set_title(ax.get_title())


if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(results_folder_ppp, "ppp_updrs_database_consistent_subjects_trem-filtered.xlsx")
    # path_to_ppp_data_reduced = os.path.join(results_folder_ppp, "ppp_updrs_database_consistent_subjects.xlsx")
    filtered_string = "TremFiltered"
    sub_folder = f"stats_summary_{filtered_string}"
    path_to_consistent_sub_responsiveness_profile = os.path.join(results_folder_ppp, f"{filtered_string}Dopamine_responsiveness_profile.xlsx")

    base_drdr_folder = "/project/3024023.01/fMRI_DRDR/"
    results_folder_drdr = "/project/3024023.01/fMRI_DRDR/updrs_analysis/"
    path_to_drdr_data = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean_complete.xlsx")

    if run_local: print("----- LOADING DATABASE PPP -----")
    excel_file_ppp = pd.ExcelFile(path_to_ppp_data_reduced)
    sheets_ppp = excel_file_ppp.sheet_names
    ppp_database = {}
    for sheet in sheets_ppp:
        ppp_database[sheet] = pd.read_excel(path_to_ppp_data_reduced, sheet_name=sheet)

    if run_local: print("----- LOADING DATABASE DRDR -----")
    excel_file_drdr = pd.ExcelFile(path_to_drdr_data)
    sheets_drdr = excel_file_drdr.sheet_names
    drdr_database = {}
    for sheet in sheets_drdr:
        if sheet in ["UPDRS OFF", "UPDRS ON"]:
            drdr_database[sheet] = pd.read_excel(path_to_drdr_data, sheet_name=sheet)
    sheets_drdr = ["UPDRS OFF", "UPDRS ON"]

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    process_drdr = False
    process_ppp = True
    ignore_unmedicated_participants_flag = False

    plot_variables_histograms_flag = False
    plot_percentage_change_flag = False
    plot_scatter_correlation_flag = False
    plot_pc_scatter_flag = False
    plot_raincloud_flag = False

    plot_pdf_flag = False
    plot_coeffs_diff_flag = False
    plot_single_sub_trends_flag = False

    create_stats_table_flag = False
    create_responsiveness_profile_flag = False
    plot_rainclouds_responsiveness_flag = True

    objDataHandlingPPP = UPDRSPlotting(ppp_database, "ppp")
    objDataHandlingDRDR = UPDRSPlotting(drdr_database, "drdr")

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if ignore_unmedicated_participants_flag:
        ppp_database = objDataHandlingPPP.remove_unmedicated_subjects()

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_variables_histograms_flag:
        if run_local: print("----- PLOTTING HISTOGRAMS -----")
        visits_ppp = [1, 2, 3]
        visits_drdr = [1]  # ["UPDRS ON", "UPDRS OFF"]
        sessions = ["off", "on"] #["off"] # ["off", "on"]
        updrs_keys_ppp = ["AvgRestTrem", "ResponseRestTrem", "AvgLimbsRestTrem", "AvgTremorUPDRS", "TremorUPDRS",
                          "PosturalTremor", "AvgBrady14Items", "AvgLimbsRigidity5Items", "TotalU3", "ResponseRestTrem",
                          "ChangeTotalU3", "ChangeTremorUPDRS", "ChangeLimbsRestTrem", "ChangeBrady14Items",
                          "ChangeLimbsRigidity5Items"]
        updrs_keys_drdr = ["AvgLimbsRestTrem", "AvgBrady14Items", "AvgLimbsRigidity5Items", ["OFFU_tot", "ONU_tot"],
                           "ChangeTotalU3", "ChangeTremorUPDRS", "ChangeLimbsRestTrem", "ChangeBrady14Items",
                           "ChangeLimbsRigidity5Items"]
        if process_ppp: objDataHandlingPPP.create_histograms(updrs_keys_ppp, visits_ppp, sessions)
        if process_drdr: objDataHandlingDRDR.create_histograms(updrs_keys_drdr, visits_drdr, sessions)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_percentage_change_flag:
        if run_local: print("----- PLOTTING PERCENTAGE CHANGE -----")
        visits_ppp = [1, 2, 3]
        visits_drdr = [1]  # ["UPDRS ON", "UPDRS OFF"]
        updrs_keys_ppp = ["AvgRestTrem", "AvgLimbsRestTrem", "AvgTremorUPDRS", "TremorUPDRS", "PosturalTremor", "AvgBrady14Items", "AvgBrady5Items", "AvgLimbsRigidity5Items", "AvgTotalU3", "TotalU3"]
        updrs_keys_drdr = ["AvgLimbsRestTrem", "AvgBrady14Items", "AvgLimbsRigidity5Items", ["OFFU_tot", "ONU_tot"]]
        if process_ppp: objDataHandlingPPP.plot_percentage_change(updrs_keys_ppp, visits_ppp, type_comparison="offon")
        if process_ppp: objDataHandlingPPP.plot_percentage_change(updrs_keys_ppp, visits_ppp, type_comparison="longitudinal")
        if process_drdr: objDataHandlingDRDR.plot_percentage_change(updrs_keys_drdr, visits_drdr, type_comparison="offon")

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_scatter_correlation_flag:
        if run_local: print("----- PLOTTING SCATTER PLOTS AND CORRELATION -----")
        updrs_conf_ppp = [
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'},
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'x': [{'off': 'LogPower', 'on': 'LogPower'}], 'y': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1], 'ses': ['off']},
            {'x': [{'off': 'LogPower', 'on': 'LogPower'}], 'y': [{'off': 'FreqPeak', 'on': 'FreqPeak'}], 'visit': [1], 'ses': ['off']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [
                {'off': 'ChangeLimbsRestTrem', 'on': 'ChangeLimbsRestTrem'},
                {'off': 'ChangeBrady14Items', 'on': 'ChangeBrady14Items'},
                {'off': 'ChangeLimbsRigidity5Items', 'on': 'ChangeLimbsRigidity5Items'},
                {'off': 'ChangeTotalU3', 'on': 'ChangeTotalU3'}
            ], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}],
             'y': [{'off': 'AvgRestTrem', 'on': 'AvgRestTrem'}, {'off': 'AvgLimbsBradyRig', 'on': 'AvgLimbsBradyRig'}],
             'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'x': [{'off': 'LogPower', 'on': 'LogPower'}],
             'y': [{'off': 'AvgRestTrem', 'on': 'AvgRestTrem'}, {'off': 'AvgLimbsBradyRig', 'on': 'AvgLimbsBradyRig'}],
             'visit': [1, 2, 3], 'ses': ['off', 'on']},
        ]
        updrs_conf_drdr = [
            {'x': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}], 'y': [{'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}], 'y': [{'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off']}
        ]

        if process_ppp: objDataHandlingPPP.plot_scatter(updrs_conf_ppp)
        if process_drdr: objDataHandlingPPP.plot_scatter(updrs_conf_drdr)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_pc_scatter_flag:
        if run_local: print("----- PLOTTING SCATTER PLOTS FOR PERCENTAGE CHANGE -----")
        updrs_conf_ppp = [
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'},
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'},
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [2], 'ses': ['off', 'on']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'},
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [3], 'ses': ['off', 'on']},
            {'x': [{'off': 'LogPower', 'on': 'LogPower'}], 'y': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'},
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'FreqPeak', 'on': 'FreqPeak'}], 'y': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'},
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'LogPower', 'on': 'LogPower'}], 'y': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}],
             'visit': [1], 'ses': ['off']},
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'ChangeLimbsBradyRig', 'on': 'ChangeLimbsBradyRig'}, {'off': 'ResponseRestTrem', 'on': 'ResponseRestTrem'}],
             'visit': [1, 2, 3], 'ses': ['off', 'on']},
        ]
        if process_ppp: objDataHandlingPPP.plot_scatter_of_percentage_change(updrs_conf_ppp)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_raincloud_flag:
        if run_local: print("----- PLOTTING RAINCLOUD PLOTS -----")
        updrs_conf_ppp = [
            {'updrs': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}, 
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'},
                {'off': 'AvgKineticTremor', 'on': 'AvgKineticTremor'},
                {'off': 'AvgPosturalTremor', 'on': 'AvgPosturalTremor'},
                {'off': 'AvgRestTrem', 'on': 'AvgRestTrem'}
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on']}
        ]
        updrs_conf_drdr = [
            {'updrs': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1], 'ses': ['off', 'on']}
        ]

        if process_ppp: objDataHandlingPPP.plot_rainclouds(updrs_conf_ppp)
        if process_drdr: objDataHandlingDRDR.plot_rainclouds(updrs_conf_drdr)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_pdf_flag:
        if run_local: print("----- PLOTTING PROBABILITY DENSITY FUNCTIONS -----")
        analysis_name = "pdf_noFilter"
        updrs_conf_ppp = [
            {'x': [
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTotalU3', 'on': 'AvgTotalU3'}
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on'], 'type': "offon"},
            {'x': [
                {'off': 'LEDD', 'on': 'LEDD'},
                {'off': 'AvgTotalU3', 'on': 'AvgTotalU3'}
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on'], 'type': "longitudinal"}
        ]
        updrs_conf_drdr = [
            {'x': [
                {'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}
            ], 'visit': [1], 'ses': ['off', 'on'], 'type': "offon"}
        ]
        if process_ppp: pdf_plots(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, style="plotly", dataset="ppp", filterHandsTremorFlag=False)
        if process_drdr: pdf_plots(drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, style="plotly", dataset="drdr", filterHandsTremorFlag=False)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_coeffs_diff_flag:
        if run_local: print("----- PLOTTING MEANS DIFFERENCE -----")
        analysis_name = "metric_diff_noTremFilter"
        updrs_conf_ppp = [
            {'updrs': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}, {'off': 'AvgTotalU3', 'on': 'AvgTotalU3'}], 'visit': [1, 2], 'ses': ['off', 'on']}
        ]
        updrs_conf_drdr = [
            {'updrs': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}, {'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off', 'on']}
        ]

        if process_ppp: coeffs_diff_plot(np.mean, "mean", ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, dataset="ppp", filterHandsTremorFlag=False)
        if process_drdr: coeffs_diff_plot(np.mean, "mean", drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, dataset="drdr", filterHandsTremorFlag=False)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_single_sub_trends_flag:
        if run_local: print("----- PLOTTING SINGLE SUBJECT TRENDS -----")
        analysis_name = "single_sub_trend_noTremFilter"
        updrs_conf_ppp = [
            {'updrs': [{'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [
                {'off': 'Brady14Items', 'on': 'Brady14Items'},
                {'off': 'LimbsRigidity5Items', 'on': 'LimbsRigidity5Items'},
                {'off': 'LimbsRestTrem', 'on': 'LimbsRestTrem'},
                {'off': 'TremorUPDRS', 'on': 'TremorUPDRS'}
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'LimbsRestTrem', 'on': 'LimbsRestTrem'},
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
        ]
        updrs_conf_drdr = [
            {'updrs': [{'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [
                {'off': 'Brady14Items', 'on': 'Brady14Items'},
                {'off': 'Sum_L_Rig_OFF', 'on': 'SUM_L_Rig_on'},
                {'off': 'UPDRS_OFF_L_RT', 'on': 'UPDRS_ON_L_RT'},
                {'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}
            ], 'visit': [1], 'ses': ['off', 'on']},
        ]

        if process_ppp: single_sub_trend_plot(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, dataset="ppp", filterHandsTremorFlag=False)
        if process_drdr: single_sub_trend_plot(drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, dataset="drdr", filterHandsTremorFlag=False)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if create_responsiveness_profile_flag:
        if run_local: print("----- CREATING RESPONSIVENESS PROFILE TABLES -----")
        # Create dopamine responsiveness participants profile table
        if process_ppp: objDataHandlingPPP.create_updrs_arbitrary_clusters()

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_rainclouds_responsiveness_flag:
        if run_local: print("----- PLOTTING RAINCLOUDS FOR RESPONSIVENESS -----")
        updrs_conf_ppp = [
            {'updrs': [
                {'off': 'ChangeTotalU3', 'on': 'ChangeTotalU3'},
                {'off': 'ChangeTremorUPDRS', 'on': 'ChangeTremorUPDRS'},
                {'off': 'ChangeLimbsRestTrem', 'on': 'ChangeLimbsRestTrem'},
                {'off': 'ChangeBrady14Items', 'on': 'ChangeBrady14Items'},
                {'off': 'ChangeLimbsRigidity5Items', 'on': 'ChangeLimbsRigidity5Items'},
                {'off': 'ResponseRestTrem', 'on': 'ResponseRestTrem'},
            ], 'visit': [1, 2, 3], 'ses': ['off']}
        ]

        if process_ppp: objDataHandlingPPP.plot_rainclouds_by_group(updrs_conf_ppp, "two-steps")
        if process_ppp: objDataHandlingPPP.plot_rainclouds_by_group(updrs_conf_ppp, "two-steps", model_labels="Labels_Model12", model_name="Model12_3clust")
        if process_ppp: objDataHandlingPPP.plot_rainclouds_by_group(updrs_conf_ppp, "two-steps", model_labels="Labels_Model11b", model_name="Model11")
        if process_ppp: objDataHandlingPPP.plot_rainclouds_by_group(updrs_conf_ppp, "arbitrary-updrs")
        if process_ppp: objDataHandlingPPP.plot_rainclouds_by_group(updrs_conf_ppp, "consistent-subjects")

    sys.exit()