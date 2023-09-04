# This script performs different initialisations of KMeans
# on the different PCA transformed data sets to find clusters and
# evaluates the clustering with adjusted rand index and sum of
# squared distances of points to the cluster centre.

# 0. Importing packages and helper functions-------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from clustering_tools import *


# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination = 0.00

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining the number of principal components
n_pca_components = 6

# Defining the number of clusters
n_clusters_list = range(2, 21, 1)

# Defining the number of kmeans initialisations
n_inits = 100

# Defining the different linkage metrics for
agg_linkage_list = ["single", "average", "complete", "ward"]

# Creating a data frame to gather the clustering results
clustering_results_master = pd.DataFrame(
    columns=[
        "model",
        "dataset",
        "scaler",
        "kmeans_init",
        "agg_linkage",
        "n_clusters",
        "weighted_ssd",
        "adj_rand_score",
    ]
)

# Defining clustering output path
clustering_overall_results_path = (
    "pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/"
)

# Agglomerative clustering dendograms
agg_clustering_dendo_path = clustering_overall_results_path + "agg_dendo_plots/"

# 2. Looping through the different data sets------------------------------------------------------

for dataset in dataset_list:
    for scaling_method in scaling_method_list:
        # Defining the data file path
        processed_data_path = (
            "pdb_af2_structural_analysis/data/processed_data/"
            + dataset
            + "/"
            + "iso_for_"
            + str(iso_for_contamination)
            + "/"
            + scaling_method
            + "/"
        )

        # Defining the pca analysis path
        pca_analysis_path = (
            "pdb_af2_structural_analysis/analysis/dim_red/pca/"
            + dataset
            + "/"
            + "iso_for_"
            + str(iso_for_contamination)
            + "/"
            + scaling_method
            + "/"
        )

        # Output path
        clustering_output_path = (
            "pdb_af2_structural_analysis/analysis/clustering/"
            + dataset
            + "/"
            + "iso_for_"
            + str(iso_for_contamination)
            + "/"
            + scaling_method
            + "/"
        )

        # Defining the path for the transformed PCA data
        pca_transformed_data_path = pca_analysis_path + "pca_transformed_data.csv"

        # Defining file paths for labels
        labels_df_path = processed_data_path + "labels.csv"

        # 3. Reading in data------------------------------------------------------------------------

        # Reading in processed data
        processed_data = pd.read_csv(
            processed_data_path + "processed_destress_data_scaled.csv"
        )

        # Reading in pca transformed data
        pca_transformed_data = pd.read_csv(pca_transformed_data_path)

        # Reading in labels
        labels_df = pd.read_csv(labels_df_path)

        # Extracting dimension columns
        dim_columns = [
            i
            for i in pca_transformed_data.columns.to_list()
            if i not in labels_df.columns.to_list()
        ]

        processed_data_joined = pd.concat(
            [
                processed_data,
                labels_df[["organism_scientific_name", "organism_group"]],
            ],
            axis=1,
        )

        # Average each principal component grouped by organism
        processed_data_avg = processed_data_joined.groupby(
            ["organism_scientific_name", "organism_group"], as_index=False
        )[processed_data.columns.to_list()].mean()

        # Average each principal component grouped by organism
        pca_transformed_data_avg = pca_transformed_data.groupby(
            ["organism_scientific_name", "organism_group"], as_index=False
        )[dim_columns].mean()

        organism_group_labels = processed_data_avg["organism_group"].to_list()

        organism_labels = processed_data_avg["organism_scientific_name"].to_list()

        processed_data_avg.drop(
            ["organism_scientific_name", "organism_group"], inplace=True, axis=1
        )

        # # Filtering PCA by the numbetr of PCA components that were selected
        # pca_transformed_data_filt = pca_transformed_data_avg.iloc[
        #     :, 2 : n_pca_components + 2
        # ]

        # 4. Running different initialisations of k means--------------------------------------------

        for n_clusters in n_clusters_list:
            for init in range(0, n_inits, 1):
                # Kmeans

                # Generating a random integer
                rand_int = np.random.randint(0, high=100000, size=1)[0]

                # Setting up kmeans
                model = KMeans(
                    n_clusters=n_clusters,
                    n_init="auto",
                    random_state=rand_int,
                )
                model_fit = model.fit(processed_data_avg)

                # Extracting the labels
                predicted_labels = model_fit.labels_

                # Extracting the sum of squared distances of samples to
                # their closest cluster centre
                weighted_ssd = model.inertia_

                # Calculating the adjusted rand score against
                # the organism labels
                adj_rand_score = metrics.adjusted_rand_score(
                    organism_group_labels,
                    predicted_labels,
                )

                # Creating a row data frame
                clustering_results = pd.DataFrame(
                    {
                        "model": "kmeans",
                        "dataset": dataset,
                        "scaler": scaling_method,
                        "agg_linkage": None,
                        "kmeans_init": init,
                        "n_clusters": n_clusters,
                        "weighted_ssd": weighted_ssd,
                        "adj_rand_score": adj_rand_score,
                    },
                    index=[0],
                )

                # Adding the hyper parameters to the data set
                clustering_results_master = pd.concat(
                    [clustering_results_master, clustering_results],
                    axis=0,
                    ignore_index=True,
                )

            # Agglomerative clustering

            for agg_linkage in agg_linkage_list:
                # Setting up agglomerative clustering
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=agg_linkage,
                    compute_distances=True,
                )
                model_fit = model.fit(processed_data_avg)

                plt.figure(figsize=(9, 8))
                plot_dendrogram(
                    model_fit,
                    truncate_mode=None,
                    labels=organism_labels,
                    orientation="left",
                    leaf_font_size=8,
                )
                plt.xticks(fontsize=10)
                plt.savefig(
                    agg_clustering_dendo_path
                    + "agg_dendo_"
                    + scaling_method
                    + "_nclusters-"
                    + str(n_clusters)
                    + "_link-"
                    + agg_linkage
                    + ".png",
                    bbox_inches="tight",
                    dpi=600,
                )
                plt.close()

                # Extracting the labels
                predicted_labels = model_fit.labels_

                # Extracting the sum of squared distances of samples to
                # their closest cluster centre
                weighted_ssd = None

                # Calculating the adjusted rand score against
                # the organism labels
                adj_rand_score = metrics.adjusted_rand_score(
                    organism_group_labels,
                    predicted_labels,
                )

                # Creating a row data frame
                clustering_results = pd.DataFrame(
                    {
                        "model": "agglomerative",
                        "dataset": dataset,
                        "scaler": scaling_method,
                        "kmeans_init": None,
                        "agg_linkage": agg_linkage,
                        "n_clusters": n_clusters,
                        "weighted_ssd": weighted_ssd,
                        "adj_rand_score": adj_rand_score,
                    },
                    index=[0],
                )

                # Adding the hyper parameters to the data set
                clustering_results_master = pd.concat(
                    [clustering_results_master, clustering_results],
                    axis=0,
                    ignore_index=True,
                )

clustering_results_master.to_csv(
    clustering_overall_results_path + "clustering_results_master_destress_6groups.csv",
    index=False,
)


kmeans_clustering_results = clustering_results_master[
    clustering_results_master["model"] == "kmeans"
].reset_index(drop=True)

agglomerative_clustering_results = clustering_results_master[
    clustering_results_master["model"] == "agglomerative"
].reset_index(drop=True)


# Plotting the average weighted_ssd and adj_rand_score by scaler and number of clusters
# for k means
clustering_results_minmax = kmeans_clustering_results[
    kmeans_clustering_results["scaler"] == "minmax"
].reset_index(drop=True)

adj_rand_ind_wssd_plot(
    data=clustering_results_minmax,
    title="KMeans " + str(n_inits) + " inits - Minmax Scaler - Organism Groups",
    file_name="kmeans_eval_minmax_destress_6groups",
    output_path=clustering_overall_results_path,
)

clustering_results_robust = kmeans_clustering_results[
    kmeans_clustering_results["scaler"] == "robust"
].reset_index(drop=True)

adj_rand_ind_wssd_plot(
    data=clustering_results_robust,
    title="KMeans " + str(n_inits) + " inits - Robust Scaler - Organism Groups",
    file_name="kmeans_eval_robust_destress_6groups",
    output_path=clustering_overall_results_path,
)

clustering_results_standard = kmeans_clustering_results[
    kmeans_clustering_results["scaler"] == "standard"
].reset_index(drop=True)

adj_rand_ind_wssd_plot(
    data=clustering_results_standard,
    title="KMeans " + str(n_inits) + " inits - Standard Scaler - Organism Groups",
    file_name="kmeans_eval_standard_destress_6groups",
    output_path=clustering_overall_results_path,
)


# Plotting the average weighted_ssd and adj_rand_score by scaler
# for agglomerative clustering

clustering_results_minmax = agglomerative_clustering_results[
    agglomerative_clustering_results["scaler"] == "minmax"
].reset_index(drop=True)

adj_rand_ind_plot(
    data=clustering_results_minmax,
    title="Agglomerative - Minmax Scaler - Organism Groups",
    hue="agg_linkage",
    file_name="agg_eval_minmax_destress_6groups",
    output_path=clustering_overall_results_path,
)

clustering_results_robust = agglomerative_clustering_results[
    agglomerative_clustering_results["scaler"] == "robust"
].reset_index(drop=True)

adj_rand_ind_plot(
    data=clustering_results_robust,
    title="Agglomerative - Robust Scaler - Organism Groups",
    hue="agg_linkage",
    file_name="agg_eval_robust_destress_6groups",
    output_path=clustering_overall_results_path,
)

clustering_results_standard = agglomerative_clustering_results[
    agglomerative_clustering_results["scaler"] == "standard"
].reset_index(drop=True)

adj_rand_ind_plot(
    data=clustering_results_standard,
    title="Agglomerative - Standard Scaler - Organism Groups",
    hue="agg_linkage",
    file_name="agg_eval_standard_destress_6groups",
    output_path=clustering_overall_results_path,
)
