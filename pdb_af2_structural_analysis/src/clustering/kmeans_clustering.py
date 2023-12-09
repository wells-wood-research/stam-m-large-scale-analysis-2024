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


# default_font = plt.rcParams["font.family"]
# print("Default font:", default_font)


# Set the font to Arial
plt.rcParams["font.family"] = "Arial"


print(plt.rcParams["font.family"])


# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination = 0.00

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining the number of principal components
n_pca_components = 2

# Defining the number of clusters
n_clusters_list = range(2, 21, 1)

# Defining the number of kmeans initialisations
n_inits = 100

# Defining the different linkage metrics for hieracrhcial clustering
linkage_list = ["single", "average", "complete", "ward"]

# Creating a data frame to gather the clustering results
clustering_results_master = pd.DataFrame(
    columns=[
        "model",
        "dataset",
        "scaler",
        "kmeans_init",
        "linkage",
        "n_clusters",
        "weighted_ssd",
        "adj_rand_score",
    ]
)

# Defining clustering output path
clustering_overall_results_path = (
    "pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/"
)

# Grouping var
group_var = "organism_group2"

if group_var == "organism_group":
    group_label = "org_group"

elif group_var == "organism_group2":
    group_label = "euk_prok"


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
            "pdb_af2_structural_analysis/analysis/clustering/"
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

        # Defining file paths for labels
        labels_df_path = processed_data_path + "labels.csv"

        # 3. Reading in data------------------------------------------------------------------------

        # Reading in processed data
        processed_data = pd.read_csv(
            processed_data_path + "processed_destress_data_scaled.csv"
        )

        # Reading in labels
        labels_df = pd.read_csv(labels_df_path)

        processed_data_joined = pd.concat(
            [
                processed_data,
                labels_df[["organism_scientific_name", group_var]],
            ],
            axis=1,
        )

        # Average each principal component grouped by organism
        processed_data_avg = processed_data_joined.groupby(
            ["organism_scientific_name", group_var], as_index=False
        )[processed_data.columns.to_list()].mean()

        organism_group_labels = processed_data_avg[group_var].to_list()

        organism_labels = processed_data_avg["organism_scientific_name"].to_list()

        labels = processed_data_avg[["organism_scientific_name", group_var]]

        processed_data_avg.drop(
            ["organism_scientific_name", group_var], inplace=True, axis=1
        )

        # Performing PCA
        var_explained_df = pca_var_explained(
            data=processed_data_avg,
            n_components=8,
            file_name="pca_var_explained_nonredund",
            output_path=pca_analysis_path,
        )

        pca_transformed_data = perform_pca(
            data=processed_data_avg,
            labels_df=labels,
            n_components=8,
            output_path=pca_analysis_path,
            file_path="pca_transformed_data",
            components_file_path="comp_contrib_nonredund_" + group_label,
        )

        sns.set_style("whitegrid")

        plt.figure(figsize=(6, 5))

        x_var_explained = var_explained_df["var_explained"][
            var_explained_df["n_components"] == 0
        ]
        y_var_explained = var_explained_df["var_explained"][
            var_explained_df["n_components"] == 1
        ]

        x_var_explained_formatted = np.round(x_var_explained.iloc[0], 2) * 100
        y_var_explained_formatted = np.round(y_var_explained.iloc[0], 2) * 100

        plot = sns.scatterplot(
            data=pca_transformed_data.sort_values(by=group_var, ascending=True),
            x="dim0",
            y="dim1",
            alpha=0.8,
            s=150,
            hue=group_var,
            legend=True,
            linewidth=0.2,
            edgecolor="black",
            palette=sns.color_palette("colorblind"),
        )
        plt.xlabel(
            "PC1 (" + str(np.int64(x_var_explained_formatted)) + "%)", fontsize=16
        )
        plt.ylabel(
            "PC2 (" + str(np.int64(y_var_explained_formatted)) + "%)", fontsize=16
        )
        # plt.title(
        #     "PCA on avg DE-STRESS metrics - "
        #     + scaling_method
        #     + " scaler \n - "
        #     + group_label,
        #     fontsize=13,
        # )
        # plt.xlim([-1.2, 1.2])
        # plt.ylim([-1, 2])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        sns.move_legend(
            plot,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncols=2,
            frameon=False,
            title="",
            # title_fontsize=16,
            fontsize=16,
        )
        plt.savefig(
            pca_analysis_path + "pca_embedding_12_nonredund_" + group_label + ".png",
            bbox_inches="tight",
            dpi=600,
        )
        plt.close()

        # Defining hiver data for plotly
        hover_data = ["organism_scientific_name", "dim0", "dim1"]

        fig = px.scatter(
            pca_transformed_data,
            x="dim0",
            y="dim1",
            color=group_var,
            opacity=0.9,
            hover_data=hover_data,
            labels={
                "dim0": "PC1",
                "dim1": "PC2",
            },
            # palette=sns.color_palette("colorblind"),
        )
        fig.update_traces(
            marker=dict(size=10, line=dict(width=0.8)),
            selector=dict(mode="markers"),
        )
        fig.write_html(
            pca_analysis_path + "pca_embedding_12_nonredund_" + group_label + ".html"
        )

        # 4. Running different initialisations of k means--------------------------------------------

        for n_clusters in n_clusters_list:
            for init in range(0, n_inits, 1):
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
                        "linkage": None,
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

            # # Agglomerative clustering

            # for linkage in linkage_list:
            #     # Setting up agglomerative clustering
            #     model = AgglomerativeClustering(
            #         n_clusters=n_clusters,
            #         linkage=linkage,
            #         compute_distances=True,
            #     )
            #     model_fit = model.fit(processed_data_avg)

            #     plt.figure(figsize=(9, 8))
            #     plot_dendrogram(
            #         model_fit,
            #         truncate_mode=None,
            #         labels=organism_labels,
            #         orientation="left",
            #         leaf_font_size=8,
            #     )
            #     plt.xticks(fontsize=10)
            #     plt.savefig(
            #         clustering_output_path
            #         + "hierarchical_dendograms_nonredund/"
            #         + "dendo_nclusters-"
            #         + str(n_clusters)
            #         + "_link-"
            #         + linkage
            #         + "_"
            #         + group_label
            #         + ".png",
            #         bbox_inches="tight",
            #         dpi=600,
            #     )
            #     plt.close()

            #     # Extracting the labels
            #     predicted_labels = model_fit.labels_

            #     # Extracting the sum of squared distances of samples to
            #     # their closest cluster centre
            #     weighted_ssd = None

            #     # Calculating the adjusted rand score against
            #     # the organism labels
            #     adj_rand_score = metrics.adjusted_rand_score(
            #         organism_group_labels,
            #         predicted_labels,
            #     )

            #     # Creating a row data frame
            #     clustering_results = pd.DataFrame(
            #         {
            #             "model": "agglomerative",
            #             "dataset": dataset,
            #             "scaler": scaling_method,
            #             "kmeans_init": None,
            #             "linkage": linkage,
            #             "n_clusters": n_clusters,
            #             "weighted_ssd": weighted_ssd,
            #             "adj_rand_score": adj_rand_score,
            #         },
            #         index=[0],
            #     )

            #     # Adding the hyper parameters to the data set
            #     clustering_results_master = pd.concat(
            #         [clustering_results_master, clustering_results],
            #         axis=0,
            #         ignore_index=True,
            #     )

clustering_results_master.to_csv(
    clustering_overall_results_path
    + "clustering_results_master_destress_nonredund_"
    + group_label
    + ".csv",
    index=False,
)


kmeans_clustering_results = clustering_results_master[
    clustering_results_master["model"] == "kmeans"
].reset_index(drop=True)

agglomerative_clustering_results = clustering_results_master[
    clustering_results_master["model"] == "agglomerative"
].reset_index(drop=True)


# Plotting the average weighted_ssd and adj_rand_score by scaler and number of clusters
# for k means and adj_rand_score by scaler and number of clusters for hierarchical clustering
for scaling_method in scaling_method_list:
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
    kmeans_clustering_results_scaler = kmeans_clustering_results[
        kmeans_clustering_results["scaler"] == scaling_method
    ].reset_index(drop=True)

    # adj_rand_ind_plot(
    #     data=kmeans_clustering_results_scaler,
    #     title="KMeans "
    #     + str(n_inits)
    #     + " inits - "
    #     + scaling_method
    #     + " scaler - "
    #     + group_label,
    #     file_name="kmeans_eval_"
    #     + scaling_method
    #     + "_destress_nonredund_"
    #     + group_label,
    #     output_path=clustering_output_path,
    # )

    plt.figure(figsize=(6, 5))

    adj_rand_ind_plot(
        data=kmeans_clustering_results_scaler,
        title="",
        file_name="kmeans_eval_"
        + scaling_method
        + "_destress_nonredund_"
        + group_label,
        # hue="group_var",
        output_path=clustering_output_path,
    )

    # agg_clustering_results_scaler = agglomerative_clustering_results[
    #     agglomerative_clustering_results["scaler"] == scaling_method
    # ].reset_index(drop=True)

    # adj_rand_ind_plot(
    #     data=agg_clustering_results_scaler,
    #     title="Hierarchical - " + scaling_method + " scaler - " + group_label,
    #     hue="linkage",
    #     file_name="hierarchical_eval_"
    #     + scaling_method
    #     + "_destress_nonredund_"
    #     + group_label,
    #     output_path=clustering_output_path,
    # )
