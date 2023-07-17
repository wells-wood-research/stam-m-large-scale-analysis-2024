# This script performs analysis on the destress data and PCA data
# to compare how the organisms are related to each other.

# 0. Importing packages----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram
from organism_phylo_tree_tools import *


# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
# iso_for_contamination_list = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
iso_for_contamination_list = [0.0]

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# 2. Looping through the different data sets------------------------------------------------------

for dataset in dataset_list:
    for iso_for_contamination in iso_for_contamination_list:
        for scaling_method in scaling_method_list:
            # Defining the data file path
            processed_data_path = (
                "data/processed_data/"
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
                "analysis/dim_red/pca/"
                + dataset
                + "/"
                + "iso_for_"
                + str(iso_for_contamination)
                + "/"
                + scaling_method
                + "/"
            )

            # Defining the path for processed AF2 DE-STRESS data
            processed_destress_data_path = (
                processed_data_path + "processed_destress_data_scaled.csv"
            )

            # Defining file paths for labels
            labels_df_path = processed_data_path + "labels.csv"

            # Defining the pca analysis path
            org_phylo_tree_analysis_path = (
                "analysis/org_phylo_tree/"
                + dataset
                + "/"
                + "iso_for_"
                + str(iso_for_contamination)
                + "/"
                + scaling_method
                + "/"
            )

            # 3. Loading in the different data sets----------------------------------------------------

            # Reading in processed destress data
            processed_destress_data = pd.read_csv(processed_destress_data_path)

            # Reading in processed destress data
            pca_transformed_data = pd.read_csv(
                pca_analysis_path + "pca_transformed_data.csv"
            )

            # Reading in labels
            labels = pd.read_csv(labels_df_path)

            # Extracting dimension columns
            dim_columns = [
                i
                for i in pca_transformed_data.columns.to_list()
                if i not in labels.columns.to_list()
            ]
            # Average each principal component grouped by organism
            pca_transformed_data_avg = pca_transformed_data.groupby(
                ["organism_scientific_name", "organism_group"], as_index=False
            )[dim_columns].mean()

            pca_transformed_data_avg_filt = pca_transformed_data_avg[
                pca_transformed_data_avg["organism_group"] != "Other"
            ]

            plot_organism_groups(
                data=pca_transformed_data_avg_filt,
                output_path=org_phylo_tree_analysis_path,
                title="",
                legend_title="Organism Group",
                # title="Dataset - "
                # + dataset
                # + ", scaling method - "
                # + scaling_method
                # + ", outlier percentage - "
                # + str(iso_for_contamination),
            )

            plot_organism_groups_plotly(
                data=pca_transformed_data_avg_filt,
                output_path=org_phylo_tree_analysis_path,
                title="",
                # title="Dataset - "
                # + dataset
                # + ", scaling method - "
                # + scaling_method
                # + ", outlier percentage - "
                # + str(iso_for_contamination),
            )

            distances_20d_all = distance_to_reference(
                data=pca_transformed_data_avg,
                dim_columns=dim_columns,
                feature="organism_scientific_name",
                distance_metric="euclidean",
                output_path=org_phylo_tree_analysis_path,
            )

            model = AgglomerativeClustering(
                distance_threshold=0,
                n_clusters=None,
                affinity="precomputed",
                linkage="single",
                compute_distances=True,
            )

            model_fitted = model.fit(distances_20d_all)

            plt.figure(figsize=(16, 8))
            # plt.title("Hierarchical Clustering Dendrogram")
            # plot the top three levels of the dendrogram
            plot_dendrogram(
                model_fitted,
                truncate_mode=None,
                labels=distances_20d_all.columns.tolist(),
                orientation="left",
                leaf_font_size=10,
            )
            plt.xticks(fontsize=11)
            plt.savefig(
                org_phylo_tree_analysis_path
                + "pca_hierarchical_clustering_dendogram.png",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close()
