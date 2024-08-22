import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy.spatial import distance
from sklearn import cluster
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap

# sys.path.insert(1, "/home/sysneu/carpue/PythonProjects/Codebase/")
# from bic import compute_bic_rss

def repeated_measures_anova_offon(database, updrs_keys, sheets, results_folder, analysis_name):
    repeated_measures = dict()
    df = pd.DataFrame()

    visits = [0, 1, 2]
    sessions = [0, 1] # 0 = OFF and 1 = ON

    visitTime = "Visit"
    condition = "Condition"
    subjects = "Subject"
    scores = "UPDRS_scores"

    visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}
    condition_labels_map = {0: "OFF", 1: "ON"}

    for key in updrs_keys:
        visit_labels = []
        condition_labels = []
        subject_labels = []
        score_values = []
        for visit in visits:
            for ses in sessions:
                updrs_scores = database[sheets[2 * visit + ses]][key]
                lengthData = len(updrs_scores)

                visit_label = visit_labels_map.get(visit, "Unknown")
                ses_label = condition_labels_map.get(ses, "Unknown")

                visit_labels.extend([visit_label] * lengthData)
                condition_labels.extend([ses_label] * lengthData)
                subject_labels.extend(database[sheets[2 * visit + ses]]["Subject"].tolist())
                score_values.extend(updrs_scores.tolist())

        df = pd.DataFrame({
            visitTime: visit_labels,
            condition: condition_labels,
            subjects: subject_labels,
            scores: score_values
        })
        # anova_results = pg.rm_anova(dv=scores, within=[visitTime, condition], subject=subjects, data=df, detailed=True)
        anova_results = AnovaRM(data=df, depvar=scores, subject=subjects, within=[visitTime, condition]).fit()
        # print(anova_results)
        repeated_measures[key] = anova_results.anova_table

    with pd.ExcelWriter(os.path.join(results_folder, f"{analysis_name}rm_anova_off-on.xlsx"), engine='openpyxl') as writer:
        for sheet_name, data in repeated_measures.items():
            data.to_excel(writer, sheet_name=sheet_name, index=True)

    return 0


def repeated_measures_mixeff_resp(database, updrs_keys, sheets, cluster_labels, results_folder, analysis_name, condition_labels_map={}):
    repeated_measures = dict()
    pairwise_results_compilation = dict()
    df = pd.DataFrame()

    visits = [0, 1, 2]

    visitTime = "Visit"
    condition = "Group"
    subjects = "Subject"
    scores = "UPDRS_scores"

    visit_labels_map = {0: "Baseline", 1: "Year 1", 2: "Year 2"}

    unique_labels = cluster_labels.unique()[cluster_labels.unique() != 0]
    if condition_labels_map == {}:
        for clust in unique_labels:
            condition_labels_map[clust] = f"Cluster{clust}"

    for key in updrs_keys:
        visit_labels = []
        condition_labels = []
        subject_labels = []
        score_values = []
        for visit in visits:
            for ses in unique_labels:
                updrs_scores = database[sheets[2 * visit]][key][cluster_labels == ses]
                lengthData = len(updrs_scores)

                visit_label = visit_labels_map.get(visit, "Unknown")
                ses_label = condition_labels_map.get(ses, "Unknown")

                visit_labels.extend([visit_label] * lengthData)
                condition_labels.extend([ses_label] * lengthData)
                subject_labels.extend(database[sheets[2 * visit]]["Subject"][cluster_labels == ses].tolist())
                score_values.extend(updrs_scores.tolist())

        df = pd.DataFrame({
            visitTime: visit_labels,
            condition: condition_labels,
            subjects: subject_labels,
            scores: score_values
        })

        # anova_results = pg.rm_anova(dv=scores, within=[visitTime, condition], subject=subjects, data=df, detailed=True)
        # anova_results = AnovaRM(data=df, depvar=scores, subject=subjects, within=[visitTime, condition]).fit()
        # model = smf.mixedlm(f"{scores} ~ {visitTime} * {condition}", df, groups=df[subjects])
        model = smf.mixedlm(f"{scores} ~ C({condition}, Treatment(reference='Responsive')) * {visitTime}", df, groups=df[subjects])
        mixedlm_results = model.fit()

        summary_df = mixedlm_results.summary().tables[1]
        repeated_measures[key] = pd.DataFrame(summary_df)

        pairwise_results = pairwise_ttests_posthoc(df, scores, visitTime, condition, subjects)
        pairwise_results_compilation[key] = pairwise_results


    with pd.ExcelWriter(os.path.join(results_folder, f"{analysis_name}_mixed_effects_resp-resi.xlsx"), engine='openpyxl') as writer:
        for sheet_name, data in repeated_measures.items():
            data.to_excel(writer, sheet_name=sheet_name, index=True)

    with pd.ExcelWriter(os.path.join(results_folder, f"{analysis_name}_posthoc_paired_T-Test.xlsx"), engine='openpyxl') as writer:
        for sheet_name, data in pairwise_results_compilation.items():
            data.to_excel(writer, sheet_name=sheet_name, index=True)
    return 0


def pairwise_ttests_posthoc(df, scores_col, visit_col, condition_col, subject_col, p_adjust_method='bonferroni'):
    all_pairwise_results = []

    # Part 1: Comparisons between clusters per visit
    for visit in df[visit_col].unique():
        visit_df = df[df[visit_col] == visit]
        pairwise_results_clusters = pg.pairwise_tests(dv=scores_col, between=condition_col, subject=subject_col, data=visit_df, padjust=p_adjust_method)
        pairwise_results_clusters[visit_col] = visit
        all_pairwise_results.append(pairwise_results_clusters)

    # Part 2: Comparisons between visits per cluster
    for cluster in df[condition_col].unique():
        cluster_df = df[df[condition_col] == cluster]
        pairwise_results_visits = pg.pairwise_tests(dv=scores_col, within=visit_col, subject=subject_col, data=cluster_df, padjust=p_adjust_method)
        pairwise_results_visits[condition_col] = cluster
        all_pairwise_results.append(pairwise_results_visits)

    # Combine all results into a single DataFrame
    final_pairwise_results_df = pd.concat(all_pairwise_results, ignore_index=True)

    return final_pairwise_results_df


def compute_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters
    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn
    X     :  multidimension np array of data points
    Returns:
    -----------------------------------------
    BIC value
    """
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X.iloc[labels == i], [centers[0][i]], 'euclidean')) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return (BIC)


def compute_bic_rss(kmeans, X):
    k = kmeans.n_clusters
    n = len(X)
    m = X.shape[1]

    # Compute RSS (Residual Sum of Squares)
    rss = np.sum(np.min(kmeans.transform(X) ** 2, axis=1))
    # BIC formula
    bic = n * np.log(rss / n) + k * np.log(n) * m
    return bic


def bic_analysis_clustering(database, max_num_clusters, results_folder):
    num_clusters = range(1, max_num_clusters+1)
    KMeans = [cluster.KMeans(n_clusters=i, init="k-means++").fit(database) for i in num_clusters]

    # now run for each cluster the BIC computation
    BIC_val = [compute_bic_rss(kmeansi, database) for kmeansi in KMeans]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, max_num_clusters+1),
        y=BIC_val,
        mode='lines+markers',
        marker=dict(size=8, color='blue'),
        line=dict(color='blue'),
        name='BIC'
    ))
    fig.update_layout(
        title='BIC by Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='BIC',
        # template='plotly_white',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(showgrid=True)
    )
    figure_name = os.path.join(results_folder, "BIC", f"bic_by_num-clusters_k-means_clustering.png")
    fig.write_image(figure_name)
    return BIC_val


def plot_grid_cluster_comparison_per_subject(series1, series2, results_folder):
    assert len(series1) == len(series2), "Series must be of the same length"
    length = len(series1)
    size = int(np.ceil(np.sqrt(length)))
    padded_length = size * size
    padded_series1 = pd.concat([series1, pd.Series([np.nan] * (padded_length - length))], ignore_index=True)
    padded_series2 = pd.concat([series2, pd.Series([np.nan] * (padded_length - length))], ignore_index=True)

    # Create a grid of size x size
    grid1 = padded_series1.values.reshape((size, size))
    grid2 = padded_series2.values.reshape((size, size))

    # Initialize a color map
    colors = ['red', 'green', 'blue', 'white']
    cmap = ListedColormap(colors)

    # Define grid colors based on the conditions
    grid_colors = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            v1 = grid1[i, j]
            v2 = grid2[i, j]
            if pd.isna(v1) or pd.isna(v2):
                grid_colors[i, j] = 3  # Handle NaN values if any
            elif v1 == 0 or v2 == 0:
                grid_colors[i, j] = 2  # Blue if at least one value is 0
            elif v1 == v2:
                grid_colors[i, j] = 0  # Red if values are the same
            else:
                grid_colors[i, j] = 1  # Green if values are different

    # Plotting
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(grid_colors, cmap=cmap, vmin=0, vmax=3)

    # Adding a horizontal color bar
    cbar = plt.colorbar(cax, ticks=[0, 1, 2, 3], orientation='horizontal')
    cbar.set_label('Grid Color Legend')
    cbar.ax.set_xticklabels(['Error', 'Match', 'Zero(no-valid)', 'NaNs'])

    # Adjust layout to fit color bar
    plt.subplots_adjust(bottom=0.15)  # Adjust this value as needed

    # Adding labels
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(np.arange(1, size + 1))
    ax.set_yticklabels(np.arange(1, size + 1))

    plt.title('Comparison of Cluster per Subject', size=16)

    figure_name = os.path.join(results_folder, "two-steps_vs_arbitrary-updrs_results_comparisons.png")
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)


def clusters_scatter_plot(data, plotting_keys, labels, clustering_method, results_folder):
    labels = np.array(labels)
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)[np.unique(labels) != 0]
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    if clustering_method == "two-steps":
        labels_dict = {1: "Responsive", 2: "Resistant"}
    else:
        labels_dict = {1: "Resistant", 2: "Responsive"}

    for i, label in enumerate(unique_labels):
        cluster_data = data[labels == label]
        ax.scatter(cluster_data[plotting_keys[0]], cluster_data[plotting_keys[1]],
                   color=colors(i), label=f'{labels_dict[label]}')

    # Set plot labels and title
    ax.set_xlabel(plotting_keys[0])
    ax.set_ylabel(plotting_keys[1])
    ax.set_title(f"Clustering with {clustering_method}")
    ax.legend()

    figure_name = os.path.join(results_folder, f"clusters_separation_{clustering_method}.png")
    plt.savefig(figure_name, bbox_inches='tight', dpi=300)
    plt.clf()
    return 0


def plot_participant_clusters(df, visitsnames, results_folder):
    colors = {0: 'blue', 2: 'red', 1: 'green'}
    num_participants = df.shape[0]
    num_cols = 10
    num_rows = int(np.ceil(num_participants / num_cols))
    figsize = (num_cols * 2, num_rows * 2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    for i, row in df.iterrows():
        visits = [row[visitsnames[0]], row[visitsnames[1]], row[visitsnames[2]]]
        colors_for_visits = [colors[val] for val in visits]
        ax = axes[i]
        bars = ax.bar([visitsnames[0], visitsnames[1], visitsnames[2]], visits, color=colors_for_visits)
        ax.set_ylim(0, 2.5)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

    for j in range(num_participants, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Clustering Distribution by Visit", fontsize=20)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    figure_name = os.path.join(results_folder, "arbitrary-updrs_clustering-by-visit.png")
    plt.savefig(figure_name, bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    run_local = True

    # Paths
    base_PPP_folder = "/project/3024023.01/PPP-POM_cohort/"
    results_folder_ppp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/"
    path_to_ppp_data_reduced = os.path.join(base_PPP_folder, "updrs_analysis", "ppp_updrs_database_consistent_subjects.xlsx")
    path_to_consistent_sub_responsiveness_profile = os.path.join(base_PPP_folder, "updrs_analysis/stats_summary_97sub/dopamine_responsiveness_profile_97.xlsx")

    bic_analysis_flag = False
    anova_off_on_flag = False
    mixed_effects_responsiv_flag = True
    analysis_name_cluster_response = "arbitraryUPDRS"
    # analysis_name_cluster_response = "two-steps-Clustering"
    plot_clustering_grid_flag = False
    plot_clusters_plots_flag = False
    
    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if run_local: print("----- LOADING DATABASE PPP -----")
    excel_file_ppp = pd.ExcelFile(path_to_ppp_data_reduced)
    sheets_ppp = excel_file_ppp.sheet_names
    ppp_database = {}
    for sheet in sheets_ppp:
        ppp_database[sheet] = pd.read_excel(path_to_ppp_data_reduced, sheet_name=sheet)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if anova_off_on_flag:
        updrs_keys = ["LimbsRestTrem", "Brady14Items", "LimbsRigidity5Items"]
        results_folder = os.path.join(results_folder_ppp, "stats_summary_97sub")
        analysis_name = ""
        repeated_measures_anova_offon(ppp_database, updrs_keys, sheets_ppp, results_folder, analysis_name)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if mixed_effects_responsiv_flag:
        updrs_keys = ["ResponseLimbsRestTrem", "ResponseBrady14Items", "ResponseLimbsRigidity5Items"]
        results_folder = os.path.join(results_folder_ppp, "stats_summary_97sub")
        if analysis_name_cluster_response == "two-steps-Clustering":
            cluster_labels = ppp_database[sheets_ppp[0]]["Predict_Model4"]
            cluster_names = {1: "Responsive", 2: "Resistant"}
        elif analysis_name_cluster_response == "arbitraryUPDRS":
            profiles97sub = pd.read_excel(path_to_consistent_sub_responsiveness_profile, sheet_name="LongitudinalSubjectProfile")
            cluster_labels = profiles97sub["Responsiveness_Baseline"] # 1=Resi and 2=Resp
            cluster_names = {1: "Resistant", 2: "Responsive"}
        repeated_measures_mixeff_resp(ppp_database, updrs_keys, sheets_ppp, cluster_labels, results_folder, analysis_name_cluster_response, cluster_names)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if bic_analysis_flag:
        results_folder = results_folder_ppp
        keys_ppp = ["ResponseTotalU3", "ResponseTremorUPDRS", "ResponseLimbsRestTrem", "ResponseBrady14Items", "ResponseLimbsRigidity5Items"]
        # keys_ppp = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]
        data_ppp = ppp_database[sheets_ppp[0]][keys_ppp]
        bic = bic_analysis_clustering(data_ppp, 15, results_folder)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_clustering_grid_flag:
        results_folder_comp = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/clustering/"
        two_steps_cluster_labels = ppp_database[sheets_ppp[0]]["Predict_Model4"]
        profiles97sub = pd.read_excel(path_to_consistent_sub_responsiveness_profile, sheet_name="LongitudinalSubjectProfile")
        arbitrary_cluster_labels = profiles97sub["Responsiveness_Baseline"]  # 1=Resi and 2=Resp
        plot_grid_cluster_comparison_per_subject(two_steps_cluster_labels, arbitrary_cluster_labels, results_folder_comp)

        visits_names = ["Responsiveness_Baseline", "Responsiveness_Year 1", "Responsiveness_Year 2"]
        df_clusters = profiles97sub[visits_names]
        plot_participant_clusters(df_clusters, visits_names, results_folder_comp)

    """ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    if plot_clusters_plots_flag:
        plotting_keys = ["ResponseTremorUPDRS", "ResponseLimbsRestTrem"]
        data_baseline = ppp_database[sheets_ppp[0]][plotting_keys]
        results_folder = "/project/3024023.01/PPP-POM_cohort/updrs_analysis/clustering/"

        two_steps_cluster_labels = ppp_database[sheets_ppp[0]]["Predict_Model4"]
        profiles97sub = pd.read_excel(path_to_consistent_sub_responsiveness_profile, sheet_name="LongitudinalSubjectProfile")
        arbitrary_cluster_labels = profiles97sub["Responsiveness_Baseline"]  # 1=Resi and 2=Resp

        clusters_scatter_plot(data_baseline, plotting_keys, two_steps_cluster_labels, "two-steps", results_folder)
        clusters_scatter_plot(data_baseline, plotting_keys, arbitrary_cluster_labels, "arbitrary-updrs", results_folder)


    sys.exit()