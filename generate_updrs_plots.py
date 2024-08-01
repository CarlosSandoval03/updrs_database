import os
import sys
import copy
from pathlib import Path

import nibabel as nib
import scipy as spy
import glob
import json
import scipy.io
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
# import ptitprince as pt
from matplotlib.ticker import PercentFormatter



def percentage_change_Elble(ratingOn, ratingOff, alpha=0.5):
    if len(ratingOn) != len(ratingOff):
        raise ValueError("Rating ON and Rating OFF have different number of elements.")
    if len(ratingOn) > 1:
        change = [100*(pow(10, alpha*(ratingOn[i] - ratingOff[i]))-1) for i, _ in enumerate(ratingOn)]
    else:
        change = 100*(pow(10, alpha*(ratingOn - ratingOff))-1)
    return [-1 * x for x in change] # [-1 * x for x in change] # change


def percentage_change_Basic(ratingOn, ratingOff, alpha=0.5):
    if len(ratingOn) != len(ratingOff):
        raise ValueError("Rating ON and Rating OFF have different number of elements.")
    if len(ratingOn) > 1:
        change = [
            100 * ((ratingOff[i] - ratingOn[i]) / ratingOff[i]) if ratingOff[i] != 0 else 0
            for i in range(len(ratingOn))
        ]
    else:
        change = 100*((ratingOff - ratingOn) / ratingOff)
    return change #[-1 * x for x in change]


def filterHandsTremorOnly(df1, df2, dataset):
    if dataset == "ppp":
        dataOff = copy.deepcopy(df1)
        dataOn = copy.deepcopy(df2)
        idxOffRUE = dataOff['U17ResTremRUE'] >= 1
        idxOffLUE = dataOff['U17ResTremLUE'] >= 1
        idxOnRUE = dataOn['U17ResTremRUE'] >= 1
        idxOnLUE = dataOn['U17ResTremLUE'] >= 1

        indexes = [((a or b) and (c or d)) for a, b, c, d in zip(idxOffRUE, idxOffLUE, idxOnRUE, idxOnLUE)]
        idx = []
        idx.append([i for i, x in enumerate(indexes) if x == True])
        idx = idx[0]
    elif dataset == "drdr":
        idx1 = df1['IncludeZach'] == 1
        idx2 = df2['IncludeZach'] == 1
        indexes = [(i == j == True) for i, j in zip(idx1, idx2)]
        idx = []
        idx.append([i for i, x in enumerate(indexes) if x == True])
        idx = idx[0]

    return idx


def create_histograms_for_pc(changes_data, updrs_key, visit, nbins, results_folder, analysis_name, style="all", typeC="offon", ses="off", includeGaussianCurve = True):
    if includeGaussianCurve:
        stat, p_value = stats.shapiro(changes_data['temp'])
        # Fit Gaussian distribution
        mu, std = np.mean(changes_data['temp']), np.std(changes_data['temp'])
        x = np.linspace(min(changes_data['temp']), max(changes_data['temp']), 100)
        p = stats.norm.pdf(x, mu, std)
        hist_values, bin_edges = np.histogram(changes_data['temp'], bins=nbins, density=True)
        p = p / max(p) * max(
            [(bin_edges[i + 1] - bin_edges[i]) * hist_values[i] for i, _ in enumerate(hist_values)]) * 100

    if style == "all" or style == "plotly":
        # fig = px.histogram(changes_data, x='temp', histnorm='percent', title='Clinical Tremor Severity', nbins=10)
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=changes_data["temp"],
            histnorm='percent',
            name='Histogram',
            xbins=dict(size=bin_edges[1] - bin_edges[0]),
            showlegend=False
        ))
        if includeGaussianCurve:
            fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Gaussian Fit', showlegend=False))
            # Add Shapiro-Wilk p-value text
            fig.add_annotation(
                x=0.95, y=0.95,
                text=f"Shapiro p- {p_value:.4f}",
                showarrow=False,
                xref='paper', yref='paper',
                xanchor='right', yanchor='top',
                font=dict(size=14, color="black")
            )
        fig.update_layout(title='Clinical Tremor Severity', xaxis_title=f'{updrs_key} (MDS-UPDRS) (% Change)',
                          yaxis_title='Density (%)')
        if typeC == "offon":
            figure_name = os.path.join(results_folder, analysis_name, f"pc_{updrs_key}_visit{visit + 1}_plotly.png")
        else:
            figure_name = os.path.join(results_folder, analysis_name, f"pc_{updrs_key}_{ses}_visits{visit[0]+1}-{visit[1]+1}_plotly.png")
        fig.write_image(figure_name)

    if style == "all" or style == "sns":
        sns.histplot(changes_data['temp'], kde=False, stat='percent', bins=nbins)

        if includeGaussianCurve:
            plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
            plt.text(0.95, 0.95, f"Shapiro p- {p_value:.4f}",
                     horizontalalignment='right', verticalalignment='top',
                     transform=plt.gca().transAxes,
                     fontsize=14, color='black')
        plt.title('Clinical Tremor Severity')
        plt.xlabel(f'{updrs_key} (MDS-UPDRS) (% Change)')
        plt.ylabel('Density (%)')
        # plt.legend()
        if typeC == "offon":
            figure_name = os.path.join(results_folder, analysis_name, f"pc_{updrs_key}_visit{visit + 1}_sns.png")
        else:
            figure_name = os.path.join(results_folder, analysis_name, f"pc_{updrs_key}_{ses}_visits{visit[0]+1}-{visit[1]+1}_sns.png")
        plt.savefig(figure_name)
        plt.clf()


def percentage_change_plot(database, updrs_keys, visits, sheets, results_folder, analysis_name, style="all", dataset="ppp", typeC="offon", filterHandsTremorFlag=True, includeGaussianCurve = True, functionPerChange = percentage_change_Elble):
    nbins = 10
    visits = [v-1 for v in visits]
    database_c = copy.deepcopy(database)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)
    for updrs_key in updrs_keys:
        if typeC == "offon":
            for visit in visits:
                if dataset == "ppp":
                    if filterHandsTremorFlag == True:
                        idx = filterHandsTremorOnly(database_c[sheets[2 * visit]], database_c[sheets[2 * visit + 1]], dataset = dataset)
                        offD = pd.Series([database_c[sheets[2 * visit]][updrs_key][i] for i in idx])
                        onD = pd.Series([database_c[sheets[2 * visit + 1]][updrs_key][i] for i in idx])
                    else:
                        offD = database_c[sheets[2 * visit]][updrs_key]
                        onD = database_c[sheets[2 * visit + 1]][updrs_key]
                elif dataset == "drdr":
                    if filterHandsTremorFlag == True:
                        idx = filterHandsTremorOnly(database_c[sheets[0]], database_c[sheets[1]], dataset = dataset)
                        if isinstance(updrs_key, list):
                            offD = pd.Series([database_c[sheets[0]][updrs_key[0]][i] for i in idx])
                            onD = pd.Series([database_c[sheets[1]][updrs_key[1]][i] for i in idx])
                        else:
                            offD = pd.Series([database_c[sheets[0]][updrs_key][i] for i in idx])
                            onD = pd.Series([database_c[sheets[1]][updrs_key][i] for i in idx])
                    else:
                        if isinstance(updrs_key, list) == True:
                            offD = database_c[sheets[0]][updrs_key[0]]
                            onD = database_c[sheets[1]][updrs_key[1]]
                        else:
                            offD = database_c[sheets[0]][updrs_key]
                            onD = database_c[sheets[1]][updrs_key]
                else:
                    raise ValueError("Select a correct dataset. Either ppp or drdr.")

                cleaned_Args = cleanNaNRowsifAnyMult(offD, onD)
                offD, onD = cleaned_Args[0], cleaned_Args[1]

                changes_data = {}
                changes_data["temp"] = functionPerChange(copy.deepcopy(onD), copy.deepcopy(offD))

                create_histograms_for_pc(changes_data, updrs_key, visit, nbins, results_folder, analysis_name, style, typeC, includeGaussianCurve)

        elif typeC == "longitudinal":
            if len(visits) <= 1:
                raise ValueError("In a longitudinal study you need to specify more than 1 session.")
            for ses in ["off", "on"]:
                if dataset == "ppp":
                    if len(visits) > 3:
                        raise ValueError("This database only have 3 visits: 1-Baseline, 2-Year1, and 3-Year 2.")
                    if filterHandsTremorFlag == True:
                        idx1 = filterHandsTremorOnly(database_c[sheets[0]], database_c[sheets[2]], dataset=dataset)
                        idx2 = filterHandsTremorOnly(database_c[sheets[0]], database_c[sheets[4]], dataset=dataset)
                        idx = set(idx1).intersection(set(idx2))
                        d1 = pd.Series([database_c[sheets[0]][updrs_key][i] for i in idx])
                        d2 = pd.Series([database_c[sheets[2]][updrs_key][i] for i in idx])
                        d3 = pd.Series([database_c[sheets[4]][updrs_key][i] for i in idx])
                    else:
                        d1 = database_c[sheets[0]][updrs_key]
                        d2 = database_c[sheets[2]][updrs_key]
                        d3 = database_c[sheets[4]][updrs_key]
                elif dataset == "drdr":
                    if len(visits) > 1:
                        raise ValueError("This database can not have longitudinal studies since there is only 1 visit.")
                else:
                    raise ValueError("Select a correct dataset. Either ppp or drdr.")

                cleaned_Args = cleanNaNRowsifAnyMult(d1, d2, d3)

                if len(visits) == 2:
                    changes_data = {}
                    changes_data["temp"] = functionPerChange(copy.deepcopy(cleaned_Args[visits[0]]), copy.deepcopy(cleaned_Args[visits[1]]))
                    create_histograms_for_pc(changes_data, updrs_key, visits, nbins, results_folder, analysis_name, style, typeC, ses, includeGaussianCurve)
                elif len(visits) == 3:
                    changes_data = {}
                    changes_data["temp"] = functionPerChange(cleaned_Args[visits[0]], cleaned_Args[visits[1]])
                    create_histograms_for_pc(changes_data, updrs_key, [visits[0], visits[1]], nbins, results_folder, analysis_name, style, typeC, ses, includeGaussianCurve)

                    changes_data = {}
                    changes_data["temp"] = functionPerChange(cleaned_Args[visits[0]], cleaned_Args[visits[2]])
                    create_histograms_for_pc(changes_data, updrs_key, [visits[0], visits[2]], nbins, results_folder, analysis_name, style, typeC, ses, includeGaussianCurve)

                    changes_data = {}
                    changes_data["temp"] = functionPerChange(cleaned_Args[visits[1]], cleaned_Args[visits[2]])
                    create_histograms_for_pc(changes_data, updrs_key, [visits[1], visits[2]], nbins, results_folder, analysis_name, style, typeC, ses, includeGaussianCurve)
    return 0


def getNamesOfComparisons(visits, sessions):
    sesName = ""
    timeName = ""
    timeNamesPos = ["basel", "year1", "year2"]
    if len(visits) == 1:
        timeName = timeNamesPos[visits[0]-1]
    elif len(visits) == 3:
        timeName = "long"
    else:
        timeName = f"{timeNamesPos[visits[0]-1]}-{timeNamesPos[visits[1]-1]}"
    if len(sessions) == 1:
        sesName = sessions[0]
    else:
        sesName = "OffOn"
    return timeName, sesName


def cleanNaNRowsifAny(xD, yD):
    dataX = copy.deepcopy(xD)
    dataY = copy.deepcopy(yD)
    idxX = dataX.isnull()
    idxY = dataY.isnull()

    indexes = [((a == True) or (b == True)) for a, b in zip(idxX, idxY)]
    idx = []
    idx.append([i for i, x in enumerate(indexes) if x == False])
    idx = idx[0]

    xD = [xD[i] for i in idx]
    yD = [yD[i] for i in idx]

    return pd.Series(xD), pd.Series(yD)


def cleanNaNRowsifAnyMult(*args):
    # Deepcopy the input series
    data_series = [copy.deepcopy(series) for series in args]

    # Create a boolean mask for non-NaN values
    mask = pd.Series([True] * len(data_series[0]))
    for series in data_series:
        mask &= ~series.isnull()

    # Filter each series based on the mask
    cleaned_series = [series[mask] for series in data_series]

    return cleaned_series


def scatter_reg_plot(database, conf_dicts, sheets, results_folder, analysis_name, style="all", dataset="ppp", filterHandsTremorFlag=True):
    # {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgTotalU3', 'on': 'AvgTotalU3'}], 'visit': [1, 2, 3], 'ses': ['off', 'on'], 'name': 'LongOffOnLEDDAvgTotalU3'}
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'black']
    df = copy.deepcopy(database)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)
    for confs in conf_dicts:
        if len(confs['x'])>1 and len(confs['x']) != len(confs['y']):
            raise ValueError("Length of X axis element should be 1 or equal to length of Y axis elements.")
        timeName, sesName = getNamesOfComparisons(confs['visit'], confs['ses'])

        fig = go.Figure()
        plt.figure(figsize=(10, 6))
        colIdx = 0

        for i_data, dataY in enumerate(confs['y']):
            if len(confs['x']) == len(confs['y']):
                dataX = confs['x'][i_data]
            else:
                dataX = confs['x'][0]
            if len(confs['y']) > 1:
                nameImageExtra = "MultipleMetrics_"
            else:
                nameImageExtra = ""
                fig = go.Figure()
                plt.figure(figsize=(10, 6))
                colIdx = 0

            for visit in confs['visit']:
                for ses in confs['ses']:
                    if ses == "off":
                        frame = 2*(visit-1)
                    elif ses == "on":
                        frame = 2*(visit-1)+1
                    else:
                        raise ValueError("Session has to be either \"off\" or \"on\". Please check the \"ses\" field in your conf difctionary.")
                    if filterHandsTremorFlag == True:
                        idx = filterHandsTremorOnly(df[sheets[frame]], df[sheets[frame]], dataset=dataset)
                        xD = pd.Series([df[sheets[frame]][dataX[ses]][i] for i in idx])
                        yD = pd.Series([df[sheets[frame]][dataY[ses]][i] for i in idx])
                    else:
                        xD = df[sheets[frame]][dataX[ses]]
                        yD = df[sheets[frame]][dataY[ses]]

                    xD, yD = cleanNaNRowsifAnyMult(xD, yD)

                    df_p = {}
                    df_p["X"] = xD
                    df_p["Y"] = yD

                    # Calculate and plot regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xD, yD)
                    x_range = np.linspace(min(xD), max(xD), 100)
                    y_range = slope * x_range + intercept
                    if dataset == "ppp":
                        if len(confs['y']) > 1:
                            label_name = f'visit{visit}-{ses}_{dataY["off"]}'
                        else:
                            label_name = f'visit{visit}-{ses}'
                    else:
                        if len(confs['y']) > 1:
                            label_name = f'{ses}_{dataY["off"]}'
                        else:
                            label_name = f'{ses}'

                    if style == "all" or style == "plotly":
                        fig.add_trace(go.Scatter(x=xD, y=yD, mode='markers', name=label_name, marker=dict(color=colors[colIdx])))
                        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color=colors[colIdx], dash='dash'), showlegend=False))
                    if style == "all" or style == "sns":
                        sns.scatterplot(data=df_p, x='X', y='Y', color=colors[colIdx], label=label_name)
                        sns.regplot(data=df_p, x='X', y='Y', scatter=False, color=colors[colIdx], line_kws={'linestyle': '-', 'linewidth': 2}, ci=None) # Add ci=None for deleting shaded area around regression lines
                        plt.title('Data Distribution')
                        plt.xlabel(dataX['off'])
                        plt.ylabel(dataY['off'])
                        plt.legend(title='Visits and Sessions')
                    colIdx = colIdx + 1
            if len(confs['y']) == 1:
                if style == "all" or style == "plotly":
                    fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title=dataY['off'])
                    if dataset == "ppp":
                        figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_plotly.png")
                    else:
                        figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_plotly.png")
                    fig.write_image(figure_name)
                if style == "all" or style == "sns":
                    if dataset == "ppp":
                        figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_sns.png")
                    else:
                        figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_sns.png")
                    plt.savefig(figure_name)
                    plt.clf()

        if len(confs['y']) > 1:
            if style == "all" or style == "plotly":
                fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title=dataY['off'])
                if dataset == "ppp":
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_plotly.png")
                else:
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_plotly.png")
                fig.write_image(figure_name)
            if style == "all" or style == "sns":
                if dataset == "ppp":
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_sns.png")
                else:
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_sns.png")
                plt.savefig(figure_name)
                plt.clf()

    return 0


def scatter_reg_plot_pChange(database, conf_dicts, sheets, results_folder, analysis_name, style="all", dataset="ppp", filterHandsTremorFlag=True, functionPerChange=percentage_change_Elble):
    # {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgTotalU3', 'on': 'AvgTotalU3'}], 'visit': [1, 2, 3], 'ses': ['off', 'on'], 'name': 'LongOffOnLEDDAvgTotalU3'}
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'black']
    df = copy.deepcopy(database)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)
    for confs in conf_dicts:
        if len(confs['x'])>1 and len(confs['x']) != len(confs['y']):
            raise ValueError("Length of X axis element should be 1 or equal to length of Y axis elements.")
        timeName, sesName = getNamesOfComparisons(confs['visit'], confs['ses'])
        fig = go.Figure()
        plt.figure(figsize=(10, 6))
        colIdx = 0

        for i_data, dataY in enumerate(confs['y']):
            if len(confs['x']) == len(confs['y']):
                dataX = confs['x'][i_data]
            else:
                dataX = confs['x'][0]
            if len(confs['y']) > 1:
                nameImageExtra = "MultipleMetrics_"
            else:
                nameImageExtra = ""
                fig = go.Figure()
                plt.figure(figsize=(10, 6))
                colIdx = 0

            for visit in confs['visit']:
                sess = confs['ses']
                frameOff = 2*(visit-1)
                frameOn = 2*(visit-1)+1
                if filterHandsTremorFlag == True:
                    idx = filterHandsTremorOnly(df[sheets[frameOff]], df[sheets[frameOff]], dataset=dataset)
                    xD = pd.Series([df[sheets[frameOff]][dataX["off"]][i] for i in idx])
                    y1 = pd.Series([df[sheets[frameOn]][dataY["off"]][i] for i in idx])
                    y2 = pd.Series([df[sheets[frameOff]][dataY["off"]][i] for i in idx])
                else:
                    xD = df[sheets[frameOff]][dataX["off"]]
                    y1 = df[sheets[frameOn]][dataY["off"]]
                    y2 = df[sheets[frameOff]][dataY["off"]]
                yD = pd.Series(functionPerChange(y1, y2))
                xD, yD = cleanNaNRowsifAnyMult(xD, yD)

                df_p = {}
                df_p["X"] = xD
                df_p["Y"] = yD

                # Calculate and plot regression line
                slope, intercept, r_value, p_value, std_err = stats.linregress(xD, yD)
                x_range = np.linspace(min(xD), max(xD), 100)
                y_range = slope * x_range + intercept
                if dataset == "ppp":
                    if len(confs['y']) > 1:
                        label_name = f'visit{visit}_{dataY["off"]}'
                    else:
                        label_name = f'visit{visit}'
                else:
                    if len(confs['y']) > 1:
                        label_name = f'{dataY["off"]}'
                    else:
                        label_name = f'Not Implemented'

                if style == "all" or style == "plotly":
                    fig.add_trace(go.Scatter(x=xD, y=yD, mode='markers', name=label_name, marker=dict(color=colors[colIdx])))
                    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', line=dict(color=colors[colIdx], dash='dash'), showlegend=False))
                if style == "all" or style == "sns":
                    sns.scatterplot(data=df_p, x='X', y='Y', color=colors[colIdx], label=label_name)
                    sns.regplot(data=df_p, x='X', y='Y', scatter=False, color=colors[colIdx], line_kws={'linestyle': '-', 'linewidth': 2}, ci=None) # Add ci=None for deleting shaded area around regression lines
                    plt.title('Data Distribution')
                    plt.xlabel(dataX['off'])
                    plt.ylabel("% Change")
                    plt.legend(title='Visits and Sessions')
                colIdx = colIdx + 1

                if len(confs['y']) == 1:
                    if style == "all" or style == "plotly":
                        fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title="% Change")
                        if dataset == "ppp":
                            figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_plotly.png")
                        else:
                            figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_plotly.png")
                        fig.write_image(figure_name)
                    if style == "all" or style == "sns":
                        if dataset == "ppp":
                            figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{timeName}_{sesName}_sns.png")
                        else:
                            figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}-{dataY['off']}_{sesName}_sns.png")
                        plt.savefig(figure_name)
                        plt.clf()

        if len(confs['y']) > 1:
            if style == "all" or style == "plotly":
                fig.update_layout(title='Data Distribution', xaxis_title=dataX['off'], yaxis_title="% Change")
                if dataset == "ppp":
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}_{timeName}_{sesName}_plotly.png")
                else:
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}_{sesName}_plotly.png")
                fig.write_image(figure_name)
            if style == "all" or style == "sns":
                if dataset == "ppp":
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}_{timeName}_{sesName}_sns.png")
                else:
                    figure_name = os.path.join(results_folder, analysis_name, f"sp_{nameImageExtra}{dataX['off']}_{sesName}_sns.png")
                plt.savefig(figure_name)
                plt.clf()
    return 0


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
                        idx = filterHandsTremorOnly(df[sheets[2*(visit-1)]], df[sheets[2*(visit-1)+1]], dataset=dataset)
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
                    if style == "all" or style == "sns":
                        sns.kdeplot(data=offD, label='OFF', color='blue', alpha=0.3, bw_adjust=0.5, fill=True)
                        sns.kdeplot(data=onD, label='ON', color='red', alpha=0.3, bw_adjust=0.5, fill=True)
                        plt.title('Probability Density Function (PDF)')
                        plt.xlabel(f"{dataX['off']}")
                        plt.ylabel('Density')
                        plt.legend(title='Sessions')
                        figure_name = os.path.join(results_folder, analysis_name, f"pdf_{dataX['off']}_OffOn_sns.png")
                        plt.savefig(figure_name)
                        plt.clf()
            elif typeC == "longitudinal":
                for ses in confs["ses"]:
                    fig = go.Figure()
                    plt.figure(figsize=(10, 6))
                    colIdx = 0
                    if filterHandsTremorFlag == True:
                        idx1 = filterHandsTremorOnly(df[sheets[0]], df[sheets[2]], dataset=dataset)
                        idx2 = filterHandsTremorOnly(df[sheets[0]], df[sheets[4]], dataset=dataset)
                        idx = set(idx1).intersection(set(idx2))
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
                    if style == "all" or style == "sns":
                        sns.kdeplot(data=d1, label='Baseline', color='blue', alpha=0.3, bw_adjust=0.5, fill=True)
                        sns.kdeplot(data=d2, label='Year 1', color='red', alpha=0.3, bw_adjust=0.5, fill=True)
                        sns.kdeplot(data=d3, label='Year 2', color='green', alpha=0.3, bw_adjust=0.5, fill=True)
                        plt.title('Probability Density Function (PDF)')
                        plt.xlabel(f"{dataX['off']}")
                        plt.ylabel('Density')
                        plt.legend(title='Sessions')
                        figure_name = os.path.join(results_folder, analysis_name, f"pdf_{dataX['off']}_long_sns.png")
                        plt.savefig(figure_name)
                        plt.clf()
            else:
                raise ValueError("Select an appropriate type of plotting PDF: OFF-ON or Longitudinal comparisons.")

    return 0


def create_kde(series, color, label):
    kde = sns.kdeplot(series, bw_adjust=0.5)
    x, y = kde.get_lines()[0].get_data()
    plt.close()
    return go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', fillcolor=color, line=dict(color=color, width=2), name=label)


def RainCloudSNS(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient="v", width_viol=.7,
                 width_box=.15, palette="Set2", bw=.2, linewidth=1, cut=0., scale="area", jitter=0.5, move=0.,
                 offset=None, point_size=3, ax=None, pointplot=False, alpha=None, dodge=False, linecolor='red', **kwargs):

    if orient == 'h':  # swap x and y
        x, y = y, x

    if ax is None:
        ax = plt.gca()

    if offset is None:
        offset = max(width_box / 1.8, .15) + .05

    # Define the properties for different plot elements
    kwcloud = {k.replace("cloud_", ""): v for k, v in kwargs.items() if k.startswith("cloud_")}
    kwbox = {k.replace("box_", ""): v for k, v in kwargs.items() if k.startswith("box_")}
    kwrain = {k.replace("rain_", ""): v for k, v in kwargs.items() if k.startswith("rain_")}
    kwpoint = {k.replace("point_", ""): v for k, v in kwargs.items() if k.startswith("point_")}

    # Draw the half-violin (cloud) plot
    sns.violinplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                   orient=orient if orient == 'v' else 'h', width=width_viol, inner=None,
                   palette=palette, bw_method=bw, linewidth=linewidth, cut=cut, density_norm=scale,
                   split=(hue is not None), ax=ax, **kwcloud)

    # Draw the boxplot (umbrella)
    sns.boxplot(x=x, y=y, hue=hue, data=data, orient=orient, width=width_box,
                order=order, hue_order=hue_order, palette=palette, dodge=dodge,
                ax=ax, **kwbox)

    # Draw the stripplot (rain)
    sns.stripplot(x=x, y=y, hue=hue, data=data, orient=orient,
                  order=order, hue_order=hue_order, palette=palette, jitter=jitter,
                  dodge=dodge, size=point_size, ax=ax, **kwrain)

    # Add pointplot (if needed)
    if pointplot:
        sns.pointplot(x=x, y=y, hue=hue, data=data, orient=orient,
                      order=order, hue_order=hue_order, dodge=width_box / 2.,
                      palette=palette if hue is not None else linecolor, ax=ax, **kwpoint)

    # Adjust alpha transparency
    if alpha is not None:
        for collection in ax.collections + ax.artists:
            collection.set_alpha(alpha)

    # Prune legend and adjust plot limits
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(labels) // (4 if pointplot else 3)], labels[:len(labels) // (4 if pointplot else 3)],
                  bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=str(hue))

    if orient == "h":
        ylim = list(ax.get_ylim())
        ylim[-1] -= (width_box + width_viol) / 4.
        ax.set_ylim(ylim)
    elif orient == "v":
        xlim = list(ax.get_xlim())
        xlim[-1] -= (width_box + width_viol) / 4.
        ax.set_xlim(xlim)

    return ax


def raincloud_plot(data, updrs_conf, sheets, results_folder, analysis_name, dataset="ppp", filterHandsTremorFlag=True):
    database = copy.deepcopy(data)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)

    for confs in updrs_conf:
        for updrs_key in confs["updrs"]:
            df = {}
            key = updrs_key["off"]
            dx = "Visit"
            dhue = "Group"

            df[key] = pd.Series(dtype='float')
            df[dx] = pd.Series(dtype='str')
            df[dhue] = pd.Series(dtype='str')

            for visit in confs["visit"]:
                for ses in confs["ses"]:
                    sheet = 2 * (visit - 1) if ses == "off" else 2 * (visit - 1) + 1
                    data_series = database[sheets[sheet]][updrs_key[ses]]

                    if filterHandsTremorFlag:
                        idx = filterHandsTremorOnly(database[sheets[sheet]], database[sheets[sheet]], dataset=dataset)
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

            cleaned = cleanNaNRowsifAnyMult(df[key], df[dx], df[dhue])
            df[key], df[dx], df[dhue] = cleaned[0], cleaned[1], cleaned[2]

            timeName, sesName = getNamesOfComparisons(confs['visit'], confs['ses'])

            f, ax = plt.subplots(figsize=(12, 5))
            ax = RainCloudSNS(x=dx, y=key, hue=dhue, data=df, palette="Set2", bw_method=0.2, linewidth=2, jitter=0.25, move=0.8, width_viol=.8,
                           ax=ax, orient="h", alpha=.7, dodge=True, pointplot=True)
            plt.title("MDS - UPDRS: Repeated Measures")
            # if style == "all" or style == "sns":
            if dataset == "ppp":
                figure_name = os.path.join(results_folder, analysis_name, f"rc_{key}_{timeName}_{sesName}_sns.png")
            else:
                figure_name = os.path.join(results_folder, analysis_name, f"rc_{updrs_key['off']}-{updrs_key['on']}_{timeName}_{sesName}_sns.png")
            plt.savefig(figure_name, bbox_inches='tight')
            plt.clf()
            plt.close()

    return 0


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
                    idx = filterHandsTremorOnly(database[sheets[sheet]], database[sheets[sheet]], dataset=dataset)

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
        plt.savefig(figure_name, bbox_inches='tight')

        plt.clf()
    return 0


def create_stats_table(dataO, results_folder, analysis_name, sheets, percentageChangeFunction=percentage_change_Elble):
    database = copy.deepcopy(dataO)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)
    stats_table = dict()
    stats_table["HandsTremorFiltered"] = pd.DataFrame()
    stats_table["Complete97(No-NaNs)"] = pd.DataFrame()
    stats_table["SampleMetrics"] = pd.DataFrame()

    # Get Idx for HandsTremorFiltered
    idx1 = filterHandsTremorOnly(database[sheets[0]], database[sheets[1]], "ppp")
    idx2 = filterHandsTremorOnly(database[sheets[2]], database[sheets[3]], "ppp")
    idx3 = filterHandsTremorOnly(database[sheets[4]], database[sheets[5]], "ppp")
    idxCommon = set(idx1).intersection(set(idx2), set(idx3))
    handsTremorFiltered = dict()

    # Set names for measurements
    stats_table["Complete97(No-NaNs)"]["Clinical ratings and quantified measurements"] = [
        "MDS-UPDRS part III",
        "MDS-UPDRS-Brad",
        "MDS-UPDRS-Rig",
        "MDS-UPDRS-Trem"
    ]
    stats_table["HandsTremorFiltered"]["Clinical ratings and quantified measurements"] = [
        "MDS-UPDRS part III",
        "MDS-UPDRS-Brad",
        "MDS-UPDRS-Rig",
        "MDS-UPDRS-Trem"
    ]

    # For the population metrics table
    stats_table["SampleMetrics"]["Metric name (first visit)"] = [
        "Age, y",
        "Sex, M/F",
        "H&Y stage, median",
        "Disease duration, y",
        "LED",
    ]
    # This scores are for participants that have data, no filtering yet
    stats_table["SampleMetrics"]["Values"] = [
        f"{np.mean(database[sheets[0]]['Age']):.3f} ({np.min(database[sheets[0]]['Age'])}-{np.max(database[sheets[0]]['Age'])})",
        f"{(database[sheets[0]]['Gender'] == 1).sum()}/{(database[sheets[0]]['Gender'] == 2).sum()}",
        f"{np.median(database[sheets[0]]['HoeYah'])} ({np.min(database[sheets[0]]['HoeYah'])}-{np.max(database[sheets[0]]['HoeYah'])})",
        f"{np.mean(database[sheets[0]]['MonthSinceDiag'])/12:.3f} ({np.min(database[sheets[0]]['MonthSinceDiag'])/12:.3f} - {np.max(database[sheets[0]]['MonthSinceDiag'])/12:.3f})",
        f"{np.mean(database[sheets[0]]['LEDD']):.3f} ({np.min(database[sheets[0]]['LEDD']):.3f}-{np.max(database[sheets[0]]['LEDD']):.3f})"
    ]
    # FOr all the other metrics table
    for sheet, data in database.items():
        handsTremorFiltered[sheet] = data.iloc[list(idxCommon)].reset_index(drop=True)

    for i, sheet in enumerate(database.keys()):
        totalsU3_C = create_string_mean_std(database[sheet]["TotalU3"])
        totalsU3_F = create_string_mean_std(handsTremorFiltered[sheet]["TotalU3"])
        brady_C = create_string_mean_std(database[sheet]["Brady14Items"])
        brady_F = create_string_mean_std(handsTremorFiltered[sheet]["Brady14Items"])
        rigi_C = create_string_mean_std(database[sheet]["LimbsRigidity5Items"])
        rigi_F = create_string_mean_std(handsTremorFiltered[sheet]["LimbsRigidity5Items"])
        tremor_C = create_string_mean_std(database[sheet]["LimbsRestTrem"])
        tremor_F = create_string_mean_std(handsTremorFiltered[sheet]["LimbsRestTrem"])
        stats_table["Complete97(No-NaNs)"][sheet] = [totalsU3_C, brady_C, rigi_C, tremor_C]
        stats_table["HandsTremorFiltered"][sheet] = [totalsU3_F, brady_F, rigi_F, tremor_F]

        if i in [1, 3, 5]:
            pcU3_C = create_string_mean_std(percentageChangeFunction(database[sheets[i]]["TotalU3"], database[sheets[i-1]]["TotalU3"], alpha=0.5))
            pcU3_F = create_string_mean_std(percentageChangeFunction(handsTremorFiltered[sheets[i]]["TotalU3"], handsTremorFiltered[sheets[i-1]]["TotalU3"], alpha=0.5))
            pcBrady_C = create_string_mean_std(percentageChangeFunction(database[sheets[i]]["Brady14Items"], database[sheets[i - 1]]["Brady14Items"], alpha=0.5))
            pcBrady_F = create_string_mean_std(percentageChangeFunction(handsTremorFiltered[sheets[i]]["Brady14Items"], handsTremorFiltered[sheets[i - 1]]["Brady14Items"], alpha=0.5))
            pcRigi_C = create_string_mean_std(percentageChangeFunction(database[sheets[i]]["LimbsRigidity5Items"], database[sheets[i - 1]]["LimbsRigidity5Items"], alpha=0.5))
            pcRigi_F = create_string_mean_std(percentageChangeFunction(handsTremorFiltered[sheets[i]]["LimbsRigidity5Items"], handsTremorFiltered[sheets[i - 1]]["LimbsRigidity5Items"], alpha=0.5))
            pcTremor_C = create_string_mean_std(percentageChangeFunction(database[sheets[i]]["LimbsRestTrem"], database[sheets[i - 1]]["LimbsRestTrem"], alpha=0.5))
            pcTremor_F = create_string_mean_std(percentageChangeFunction(handsTremorFiltered[sheets[i]]["LimbsRestTrem"], handsTremorFiltered[sheets[i - 1]]["LimbsRestTrem"], alpha=0.5))
            stats_table["Complete97(No-NaNs)"][f"% Change {i}"] = [pcU3_C, pcBrady_C, pcRigi_C, pcTremor_C]
            stats_table["HandsTremorFiltered"][f"% Change {i}"] = [pcU3_F, pcBrady_F, pcRigi_F, pcTremor_F]

            # Mean diff and CI
            u3C = create_string_meandiff_ci(database[sheets[i]]["TotalU3"], database[sheets[i-1]]["TotalU3"])
            u3F = create_string_meandiff_ci(handsTremorFiltered[sheets[i]]["TotalU3"], handsTremorFiltered[sheets[i-1]]["TotalU3"])
            bradyC = create_string_meandiff_ci(database[sheets[i]]["Brady14Items"], database[sheets[i - 1]]["Brady14Items"])
            bradyF = create_string_meandiff_ci(handsTremorFiltered[sheets[i]]["Brady14Items"], handsTremorFiltered[sheets[i - 1]]["Brady14Items"])
            rigiC = create_string_meandiff_ci(database[sheets[i]]["LimbsRigidity5Items"], database[sheets[i - 1]]["LimbsRigidity5Items"])
            rigiF = create_string_meandiff_ci(handsTremorFiltered[sheets[i]]["LimbsRigidity5Items"], handsTremorFiltered[sheets[i - 1]]["LimbsRigidity5Items"])
            tremorC = create_string_meandiff_ci(database[sheets[i]]["LimbsRestTrem"], database[sheets[i - 1]]["LimbsRestTrem"])
            tremorF = create_string_meandiff_ci(handsTremorFiltered[sheets[i]]["LimbsRestTrem"], handsTremorFiltered[sheets[i - 1]]["LimbsRestTrem"])
            stats_table["Complete97(No-NaNs)"][f"MeansDiff (CI) {i}"] = [u3C, bradyC, rigiC, tremorC]
            stats_table["HandsTremorFiltered"][f"MeansDiff (CI) {i}"] = [u3F, bradyF, rigiF, tremorF]

            # Cohen's d and CI
            u3C = create_string_cohen_d_ci(database[sheets[i]]["TotalU3"], database[sheets[i - 1]]["TotalU3"])
            u3F = create_string_cohen_d_ci(handsTremorFiltered[sheets[i]]["TotalU3"], handsTremorFiltered[sheets[i - 1]]["TotalU3"])
            bradyC = create_string_cohen_d_ci(database[sheets[i]]["Brady14Items"], database[sheets[i - 1]]["Brady14Items"])
            bradyF = create_string_cohen_d_ci(handsTremorFiltered[sheets[i]]["Brady14Items"], handsTremorFiltered[sheets[i - 1]]["Brady14Items"])
            rigiC = create_string_cohen_d_ci(database[sheets[i]]["LimbsRigidity5Items"], database[sheets[i - 1]]["LimbsRigidity5Items"])
            rigiF = create_string_cohen_d_ci(handsTremorFiltered[sheets[i]]["LimbsRigidity5Items"], handsTremorFiltered[sheets[i - 1]]["LimbsRigidity5Items"])
            tremorC = create_string_cohen_d_ci(database[sheets[i]]["LimbsRestTrem"], database[sheets[i - 1]]["LimbsRestTrem"])
            tremorF = create_string_cohen_d_ci(handsTremorFiltered[sheets[i]]["LimbsRestTrem"], handsTremorFiltered[sheets[i - 1]]["LimbsRestTrem"])
            stats_table["Complete97(No-NaNs)"][f"Cohen's d (CI) {i}"] = [u3C, bradyC, rigiC, tremorC]
            stats_table["HandsTremorFiltered"][f"Cohen's d (CI) {i}"] = [u3F, bradyF, rigiF, tremorF]

            # p-value
            u3C = paired_ttest(database[sheets[i]]["TotalU3"], database[sheets[i - 1]]["TotalU3"])
            u3F = paired_ttest(handsTremorFiltered[sheets[i]]["TotalU3"], handsTremorFiltered[sheets[i - 1]]["TotalU3"])
            bradyC = paired_ttest(database[sheets[i]]["Brady14Items"], database[sheets[i - 1]]["Brady14Items"])
            bradyF = paired_ttest(handsTremorFiltered[sheets[i]]["Brady14Items"], handsTremorFiltered[sheets[i - 1]]["Brady14Items"])
            rigiC = paired_ttest(database[sheets[i]]["LimbsRigidity5Items"], database[sheets[i - 1]]["LimbsRigidity5Items"])
            rigiF = paired_ttest(handsTremorFiltered[sheets[i]]["LimbsRigidity5Items"], handsTremorFiltered[sheets[i - 1]]["LimbsRigidity5Items"])
            tremorC = paired_ttest(database[sheets[i]]["LimbsRestTrem"], database[sheets[i - 1]]["LimbsRestTrem"])
            tremorF = paired_ttest(handsTremorFiltered[sheets[i]]["LimbsRestTrem"], handsTremorFiltered[sheets[i - 1]]["LimbsRestTrem"])
            stats_table["Complete97(No-NaNs)"][f"p-value {i}"] = [u3C, bradyC, rigiC, tremorC]
            stats_table["HandsTremorFiltered"][f"p-value {i}"] = [u3F, bradyF, rigiF, tremorF]

    with pd.ExcelWriter(os.path.join(results_folder, analysis_name, "table_summary_stats.xlsx"), engine='openpyxl') as writer:
        for sheet_name, data in stats_table.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    return 0


def create_string_mean_std(data):
    meanKey = np.mean(data)
    stdKey = np.std(data)
    sumStr = f"{meanKey:.3f}\u00B1{stdKey:.3f}"
    return sumStr


def create_string_meandiff_ci_array(data1, data2):
    differences = [data1[i] - data2[i] for i, _ in enumerate(data1)]
    meanDiff = np.mean(differences)
    meanDiff = abs(meanDiff)
    stdDiff = np.std(differences)
    n = len(data1)
    df = n - 1
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_critical = t.ppf(1 - alpha / 2, df)
    margin_of_error = t_critical * (stdDiff / np.sqrt(n))
    lower_bound = meanDiff - margin_of_error
    upper_bound = meanDiff + margin_of_error
    sumStr = f"{meanDiff:.3f} ({lower_bound:.3f}-{upper_bound:.3f})"
    return sumStr


def create_string_meandiff_ci(data1, data2):
    mean1, mean2 = np.mean(data1), np.mean(data2)
    mean_diff = mean1 - mean2
    mean_diff = abs(mean_diff)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    se_diff = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
    confidence_level = 0.95
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    margin_error = z * se_diff
    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error

    sumStr = f"{mean_diff:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
    return sumStr


def cohens_d(data1, data2):
    n1 = len(data1)
    n2 = len(data2)
    dof = n1 + n2 - 2
    std_pooled = np.sqrt((((n1 - 1)*(np.std(data1)**2))+((n2 - 1)*(np.std(data2)**2))) / dof)
    d = abs(np.mean(data1) - np.mean(data2)) / std_pooled
    J = hedges_correction(dof)
    return J*d


def hedges_correction(df):
    if df == 0:
        return np.nan
    else:
        J = np.exp(gammaln(df / 2) - np.log(np.sqrt(df / 2)) - gammaln((df - 1) / 2))
        return J


def create_string_cohen_d_ci(data1, data2):
    z = 1.96 # Critical value z for a 95% confidence
    confidence_level = 0.95
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    n1 = len(data1)
    n2 = len(data2)
    d = cohens_d(data1, data2)
    std_error_d = np.sqrt(((n1 + n2) / (n1 * n2)) + ((d ** 2)/(2 * (n1 + n2))))
    margin_error = z * std_error_d
    lower_bound = d - margin_error
    upper_bound = d + margin_error
    sumStr = f"{d:.3f} ({lower_bound:.3f}-{upper_bound:.3f})"
    return sumStr


def paired_ttest(data1, data2):
    t_stat, p_value = stats.ttest_rel(data2, data1)
    return f"{p_value:.5f}"


def single_sub_trend_plot(data, updrs_conf, sheets, results_folder, analysis_name, dataset="ppp", filterHandsTremorFlag=True):
    # Instead of group I have sub-ID
    database = copy.deepcopy(data)
    Path(os.path.join(results_folder, analysis_name)).mkdir(parents=True, exist_ok=True)

    subIDKey = "Subject" if dataset == "ppp" else "PatCode"
    dx = "Visit"
    dhue = subIDKey

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
            sheetOFF = 2 * (visit - 1) if dataset == "ppp" else 0
            sheetON = 2 * (visit - 1) + 1 if dataset == "ppp" else 1
            if filterHandsTremorFlag:
                idx = filterHandsTremorOnly(database[sheets[sheetON]], database[sheets[sheetOFF]], dataset=dataset)

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

        df_melted = df.melt(id_vars=[dx, dhue], var_name='UPDRS', value_name='value')
        g = sns.FacetGrid(df_melted, col='UPDRS', hue=dhue, sharey=False, height=5, aspect=1.5)
        g.map_dataframe(plot_individual_lines, dx=dx, dhue=dhue)
        # g.add_legend(title=dhue)
        g.set_axis_labels(dx, "UPDRS Score")
        g.set_titles("{col_name}")

        # Perform ANOVA for each measurement
        # for measurement in df_melted['UPDRS'].unique():
        #     formula = f'value ~ C({dx}) + C({dhue}) + C({dx}):C({dhue})'
        #     model = ols(formula, data=df_melted[df_melted['UPDRS'] == measurement]).fit()
        #     anova_table = sm.stats.anova_lm(model, typ=2)
        #     anova_table.to_excel(os.path.join(results_folder, analysis_name, f"ANOVA results for {measurement}_{timeName}-{sesName}.xlsx"), engine='openpyxl', index=True)

        if dataset == "ppp":
            figure_name = os.path.join(results_folder, analysis_name, f"cd_{nameUPDRSkeys}_{timeName}-{sesName}_sns.png")
        else:
            figure_name = os.path.join(results_folder, analysis_name, f"cd_{nameUPDRSkeys}_{sesName}_sns.png")
        plt.savefig(figure_name, bbox_inches='tight')

        plt.clf()
    return 0


def plot_individual_lines(data, dx, dhue, **kwargs):
    participants = data[dhue].unique()
    for participant in participants:
        sns.lineplot(data=data[data[dhue] == participant], x=dx, y='value', errorbar=None, marker='o', **kwargs)


if __name__ == "__main__":
    # General variables
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")

    base_drdr_folder = "/project/3024023.01/fMRI_DRDR/"
    results_folder_drdr = "/project/3024023.01/fMRI_DRDR/updrs_analysis/"
    path_to_drdr_data = os.path.join(base_drdr_folder, "updrs_analysis", "DRDR_Results_clean.xlsx")

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
    plot_percentage_change_flag = False
    plot_scatter_correlation_flag = False
    plot_raincloud_flag = False
    plot_pdf_flag = False
    plot_coeffs_diff_flag = False
    create_stats_table_flag = True
    plot_single_sub_trends_flag = False

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_percentage_change_flag:
        if run_local: print("----- PLOTTING PERCENTAGE CHANGE -----")
        analysis_name = "Percentage-Change-Basic_noFilter"
        visits_ppp = [1, 2, 3]
        visits_drdr = [1]  # ["UPDRS ON", "UPDRS OFF"]
        updrs_keys_ppp = ["AvgLimbsRestTrem", "AvgTremorUPDRS", "TremorUPDRS", "PosturalTremor", "AvgBrady14Items", "AvgBrady5Items", "AvgLimbsRigidity5Items", "AvgLimbsRigidity4Items", "AvgTotalU3", "TotalU3"]
        updrs_keys_drdr = ["AvgLimbsRestTrem", "AvgBrady14Items", "AvgLimbsRigidity", ["OFFU_tot", "ONU_tot"]]

        percentage_change_plot(ppp_database, updrs_keys_ppp, visits_ppp, sheets_ppp, results_folder_ppp, analysis_name, style="plotly", dataset="ppp", typeC="offon", filterHandsTremorFlag=False, includeGaussianCurve = True, functionPerChange = percentage_change_Basic)
        percentage_change_plot(ppp_database, updrs_keys_ppp, visits_ppp, sheets_ppp, results_folder_ppp, analysis_name, style="plotly", dataset="ppp", typeC="longitudinal", filterHandsTremorFlag=False, includeGaussianCurve=True, functionPerChange = percentage_change_Basic)
        percentage_change_plot(drdr_database, updrs_keys_drdr, visits_drdr, sheets_drdr, results_folder_drdr, analysis_name, style="plotly", dataset="drdr", typeC="offon", filterHandsTremorFlag=False, includeGaussianCurve = True, functionPerChange = percentage_change_Basic)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_scatter_correlation_flag:
        if run_local: print("----- PLOTTING SCATTER PLOTS AND CORRELATION -----")
        analysis_name = "scatter_plots_Filter"
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
            {'x': [{'off': 'LEDD', 'on': 'LEDD'}], 'y': [{'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']}
        ]
        updrs_conf_drdr = [
            {'x': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}], 'y': [{'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off', 'on']},
            {'x': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}], 'y': [{'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off']}
        ]

        scatter_reg_plot(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, style="plotly", dataset="ppp", filterHandsTremorFlag=True)
        scatter_reg_plot(drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, style="plotly", dataset="drdr", filterHandsTremorFlag=True)

        analysis_name = "scatter_plots_Basic_noFilter"
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
            ], 'visit': [3], 'ses': ['off', 'on']}
        ]
        scatter_reg_plot_pChange(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, style="plotly", dataset="ppp", filterHandsTremorFlag=False, functionPerChange=percentage_change_Basic)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_raincloud_flag:
        if run_local: print("----- PLOTTING RAINCLOUD PLOTS -----")
        analysis_name = "raincloud_plots_Filter"
        updrs_conf_ppp = [
            {'updrs': [
                {'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}, 
                {'off': 'AvgLimbsRigidity5Items', 'on': 'AvgLimbsRigidity5Items'},
                {'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'},
                {'off': 'AvgTremorUPDRS', 'on': 'AvgTremorUPDRS'}
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on']}
        ]
        updrs_conf_drdr = [
            {'updrs': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}, {'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off', 'on']}
        ]

        raincloud_plot(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, dataset="ppp", filterHandsTremorFlag=True)
        raincloud_plot(drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, dataset="drdr", filterHandsTremorFlag=True)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_pdf_flag:
        if run_local: print("----- PLOTTING PROBABILITY DENSITY FUNCTIONS -----")
        analysis_name = "pdf"
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
        pdf_plots(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, style="plotly", dataset="ppp", filterHandsTremorFlag=True)
        pdf_plots(drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, style="plotly", dataset="drdr", filterHandsTremorFlag=True)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_coeffs_diff_flag:
        if run_local: print("----- PLOTTING MEANS DIFFERENCE -----")
        analysis_name = "metric_diff_plots"
        updrs_conf_ppp = [
            {'updrs': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1, 2, 3], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}, {'off': 'AvgTotalU3', 'on': 'AvgTotalU3'}], 'visit': [1, 2], 'ses': ['off', 'on']}
        ]
        updrs_conf_drdr = [
            {'updrs': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}, {'off': 'OFFU_tot', 'on': 'ONU_tot'}], 'visit': [1], 'ses': ['off', 'on']}
        ]

        coeffs_diff_plot(np.mean, "mean", ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, dataset="ppp", filterHandsTremorFlag=True)
        # coeffs_diff_plot(np.mean, "mean", drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, dataset="drdr", filterHandsTremorFlag=True)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if create_stats_table_flag:
        if run_local: print("----- CREATING TABLE WITH STATS SUMMARY PPP -----")
        analysis_name = "stats_summary"
        create_stats_table(ppp_database, results_folder_ppp, analysis_name, sheets_ppp, percentageChangeFunction=percentage_change_Basic)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_single_sub_trends_flag:
        if run_local: print("----- PLOTTING MEANS DIFFERENCE -----")
        analysis_name = "single_sub_trend"
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
            ], 'visit': [1, 2, 3], 'ses': ['off', 'on']}
        ]
        updrs_conf_drdr = [
            {'updrs': [{'off': 'AvgBrady14Items', 'on': 'AvgBrady14Items'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgLimbsRigidity', 'on': 'AvgLimbsRigidity'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'AvgLimbsRestTrem', 'on': 'AvgLimbsRestTrem'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [{'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}], 'visit': [1], 'ses': ['off', 'on']},
            {'updrs': [
                {'off': 'Brady14Items', 'on': 'Brady14Items'},
                {'off': 'Sum_L_Rig_OFF', 'on': 'SUM_L_Rig_on'},
                {'off': 'UPDRS_OFF_L_RT', 'on': 'UPDRS_ON_L_RT'},
                {'off': 'OFFTremorUPDRS', 'on': 'ONTremorUPDRS'}
            ], 'visit': [1], 'ses': ['off', 'on']},
        ]

        single_sub_trend_plot(ppp_database, updrs_conf_ppp, sheets_ppp, results_folder_ppp, analysis_name, dataset="ppp", filterHandsTremorFlag=False)
        single_sub_trend_plot(drdr_database, updrs_conf_drdr, sheets_drdr, results_folder_drdr, analysis_name, dataset="drdr", filterHandsTremorFlag=False)