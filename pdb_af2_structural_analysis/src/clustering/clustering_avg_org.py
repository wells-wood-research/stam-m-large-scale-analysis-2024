# This script performs KMeans on the different PCA transformed data sets
# to find clusters and evaluating the clusters against organisms

# 0. Importing packages and helper functions---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination = 0.00

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]
# scaling_method_list = ["robust"]

# Defining the number of principal components
n_pca_components_list = range(2, 8, 1)
# n_pca_components_list = [8]

# Defining the number of clusters
n_clusters_list = range(2, 10, 1)
# n_clusters_list = [6]

# # Defining the clustering method list
clustering_method_list = ["kmeans"]

# Spectral parameters
spectral_affinity_list = ["rbf"]
# spectral_n_neighbors_list = range(2, 10, 1)
spectral_rbf_gamma_list = [0.1, 0.5, 1, 1.5, 2, 2.5, 3]
spectral_assign_labels_list = ["kmeans", "discretize", "cluster_qr"]


# Creating a data frame to gather the clustering results
clustering_results_master = pd.DataFrame(
    columns=[
        "model_id",
        "adj_rand_score",
    ]
)


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

        # Average each principal component grouped by organism
        pca_transformed_data_avg = pca_transformed_data.groupby(
            ["organism_scientific_name", "organism_group"], as_index=False
        )[dim_columns].mean()

        organism_labels = pca_transformed_data_avg["organism_group"].to_list()

        # pca_transformed_data_avg.drop(
        #     ["organism_scientific_name", "organism_group"], inplace=True, axis=1
        # )

        # pca_transformed_data_avg = pca_transformed_data[dim_columns]

        # organism_labels = pca_transformed_data["organism_group"].to_list()

        # neighbors = NearestNeighbors(n_neighbors=50)
        # neighbors_fit = neighbors.fit(pca_transformed_data_filt)
        # distances, indices = neighbors_fit.kneighbors(pca_transformed_data_filt)

        # distances = np.sort(distances, axis=0)
        # distances = distances[:, 1]
        # plt.plot(distances)
        for n_pca_components in n_pca_components_list:
            pca_transformed_data_filt = pca_transformed_data_avg.iloc[
                :, 2 : n_pca_components + 2
            ]
            for clustering_method in clustering_method_list:
                if clustering_method in ["kmeans", "gaussmix"]:
                    for n_clusters in n_clusters_list:
                        # Creating a model id
                        model_id = (
                            "pca_comp-"
                            + str(n_pca_components)
                            + "_"
                            + clustering_method
                            + "_"
                            + dataset
                            + "_"
                            + scaling_method
                            + "_clusters-"
                            + str(n_clusters)
                        )

                        # Fitting the model
                        if clustering_method == "kmeans":
                            model = KMeans(
                                n_clusters=n_clusters,
                                random_state=42,
                                n_init="auto",
                            )
                            model_fit = model.fit(pca_transformed_data_filt)

                            # Extracting the labels
                            predicted_labels = model_fit.labels_

                        elif clustering_method == "gaussmix":
                            model = GaussianMixture(
                                n_components=n_clusters, random_state=42
                            ).fit(pca_transformed_data_filt)

                            predicted_labels = model.predict(pca_transformed_data_filt)

                        adj_rand_score = metrics.adjusted_rand_score(
                            organism_labels,
                            predicted_labels,
                        )

                        # Creating a data frame
                        predicted_labels_df = pd.DataFrame(
                            predicted_labels, columns=["pred_labels"]
                        )

                        # Saving dataset
                        predicted_labels_df.to_csv(
                            clustering_output_path + model_id + "_pred_df.csv"
                        )

                        # Joining onto the pca transformed data
                        kmeans_predicted_clusters_df = pd.concat(
                            [pca_transformed_data_avg, predicted_labels_df], axis=1
                        )

                        sns.set_style("whitegrid")

                        # PCA 2d scatter plot
                        plot = sns.scatterplot(
                            x="dim0",
                            y="dim1",
                            data=kmeans_predicted_clusters_df.sort_values(
                                by="organism_group", ascending=True
                            ),
                            hue="pred_labels",
                            # hue_order=hue_order,
                            palette=sns.color_palette(
                                "colorblind",
                                as_cmap=True,
                            ),
                            style="organism_group",
                            alpha=0.8,
                            s=150,
                            legend=True,
                        )
                        plt.xlabel("PC1", fontsize=14)
                        plt.ylabel("PC2", fontsize=14)
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=14)
                        # plt.title(title)

                        # legend = plt.gca().get_legend()
                        # handles, labels = legend.legendHandles, [
                        #     text.get_text() for text in legend.get_texts()
                        # ]
                        # labels_new = [
                        #     "Clusters",
                        #     "1",
                        #     "2",
                        #     "3",
                        #     "4",
                        #     "5",
                        #     "6",
                        #     "Organism Group",
                        #     "Animal",
                        #     "Archaea",
                        #     "Bacteria",
                        #     "Fungi",
                        #     "Plant",
                        #     "Protozoan",
                        # ]
                        sns.move_legend(
                            plot,
                            # handles=handles,
                            # labels=labels_new,
                            loc="upper left",
                            bbox_to_anchor=(1, 1),
                            frameon=False,
                            fontsize=12,
                        )
                        # plot.get_legend().set_title("K-Means Clusters")
                        plt.savefig(
                            clustering_output_path + model_id + "_12.png",
                            bbox_inches="tight",
                            dpi=300,
                        )
                        plt.close()

                        # # PCA 2d scatter plot
                        # plot = sns.scatterplot(
                        #     x="dim1",
                        #     y="dim2",
                        #     data=kmeans_predicted_clusters_df,
                        #     hue="pred_labels",
                        #     # hue_order=hue_order,
                        #     palette=sns.color_palette("colorblind", n_colors=6),
                        #     style="organism_group",
                        #     alpha=0.9,
                        #     s=100,
                        #     legend=True,
                        # )
                        # plt.xlabel("PC2")
                        # plt.ylabel("PC3")
                        # # plt.title(title)
                        # sns.move_legend(
                        #     plot, "upper left", bbox_to_anchor=(1, 1), frameon=False
                        # )
                        # plot.get_legend().set_title("K-Means Clusters")
                        # plt.savefig(
                        #     clustering_output_path + model_id + "_23.png",
                        #     bbox_inches="tight",
                        #     dpi=300,
                        # )
                        # plt.close()

                        # # PCA 2d scatter plot
                        # plot = sns.scatterplot(
                        #     x="dim0",
                        #     y="dim2",
                        #     data=kmeans_predicted_clusters_df,
                        #     hue="pred_labels",
                        #     # hue_order=hue_order,
                        #     palette=sns.color_palette("colorblind", n_colors=6),
                        #     style="organism_group",
                        #     alpha=0.9,
                        #     s=100,
                        #     legend=True,
                        # )
                        # plt.xlabel("PC1")
                        # plt.ylabel("PC3")
                        # # plt.title(title)
                        # sns.move_legend(
                        #     plot, "upper left", bbox_to_anchor=(1, 1), frameon=False
                        # )
                        # plot.get_legend().set_title("K-Means Clusters")
                        # plt.savefig(
                        #     clustering_output_path + model_id + "_13.png",
                        #     bbox_inches="tight",
                        #     dpi=300,
                        # )
                        # plt.close()

                        # fig = px.scatter_3d(
                        #     kmeans_predicted_clusters_df,
                        #     x="dim0",
                        #     y="dim1",
                        #     z="dim2",
                        #     color="pred_labels",
                        #     color_discrete_sequence=sns.color_palette(
                        #         "colorblind",
                        #         as_cmap=True,
                        #         n_colors=6,
                        #     ),
                        #     symbol="organism_group",
                        #     symbol_sequence=[
                        #         "circle",
                        #         "square",
                        #         "diamond",
                        #         "cross",
                        #         "square-open",
                        #         "circle-open",
                        #     ],
                        #     hover_data=[
                        #         "organism_scientific_name",
                        #         "organism_group",
                        #         "dim0",
                        #         "dim1",
                        #     ],
                        #     # title=title,
                        #     opacity=0.8,
                        # )
                        # fig.update_traces(marker_size=12)
                        # fig.update_layout(showlegend=False)
                        # fig.write_html(
                        #     clustering_output_path + model_id + ".html",
                        # )

                        # Creating a row data frame
                        clustering_results = pd.DataFrame(
                            {
                                "model_id": model_id,
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

                        print(model_id)

                elif clustering_method in ["spectral"]:
                    for n_clusters in n_clusters_list:
                        for spectral_affinity in spectral_affinity_list:
                            for spectral_rbf_gamma in spectral_rbf_gamma_list:
                                for (
                                    spectral_assign_labels
                                ) in spectral_assign_labels_list:
                                    # Creating a model id
                                    model_id = (
                                        "pca_comp-"
                                        + str(n_pca_components)
                                        + "_"
                                        + clustering_method
                                        + "_"
                                        + dataset
                                        + "_"
                                        + scaling_method
                                        + "_nclusters-"
                                        + str(n_clusters)
                                        + "_affinity-"
                                        + spectral_affinity
                                        + "_gamma-"
                                        + str(spectral_rbf_gamma)
                                        + "_assignlabels-"
                                        + spectral_assign_labels
                                    )

                                    model = SpectralClustering(
                                        n_clusters=n_clusters,
                                        assign_labels=spectral_assign_labels,
                                        affinity=spectral_affinity,
                                        gamma=spectral_rbf_gamma,
                                        random_state=42,
                                    ).fit(pca_transformed_data_filt)

                                    predicted_labels = model.labels_

                                    adj_rand_score = metrics.adjusted_rand_score(
                                        organism_labels,
                                        predicted_labels,
                                    )

                                    # Creating a row data frame
                                    clustering_results = pd.DataFrame(
                                        {
                                            "model_id": model_id,
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

                                    print(model_id)

                # elif clustering_method in ["dbscan"]:
                #     for dbscan_eps in dbscan_eps_list:
                #         for dbscan_min_samples in dbscan_min_samples_list:
                #             # Creating a model id
                #             model_id = (
                #                 "pca_comp_"
                #                 + str(n_pca_components)
                #                 + "_"
                #                 + clustering_method
                #                 + "_"
                #                 + dataset
                #                 + "_"
                #                 + scaling_method
                #                 + "_eps"
                #                 + str(dbscan_eps)
                #                 + "_minsamples"
                #                 + str(dbscan_min_samples)
                #             )

                #             model = DBSCAN(
                #                 eps=dbscan_eps, min_samples=dbscan_min_samples
                #             ).fit(pca_transformed_data_filt)
                #             predicted_labels = model.labels_

                #             adj_rand_score = metrics.adjusted_rand_score(
                #                 organism_labels,
                #                 predicted_labels,
                #             )

                #             # Creating a row data frame
                #             clustering_results = pd.DataFrame(
                #                 {
                #                     "model_id": model_id,
                #                     "adj_rand_score": adj_rand_score,
                #                 },
                #                 index=[0],
                #             )

                #             # Adding the hyper parameters to the data set
                #             clustering_results_master = pd.concat(
                #                 [clustering_results_master, clustering_results],
                #                 axis=0,
                #                 ignore_index=True,
                #             )

                #             print(model_id)

                # # Creating a data frame
                # predicted_labels_df = pd.DataFrame(
                #     predicted_labels, columns=["kmeans_pred_labels"]
                # )

                # # Joining onto the pca transformed data
                # kmeans_predicted_clusters_df = pd.concat(
                #     [pca_transformed_data, predicted_labels_df], axis=1
                # )

                # # Saving dataset
                # kmeans_predicted_clusters_df.to_csv(
                #     clustering_output_path + "kmeans_predicted_clusters_df.csv"
                # )


clustering_results_master.to_csv(
    "pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/clustering_results_master.csv",
    index=False,
)
