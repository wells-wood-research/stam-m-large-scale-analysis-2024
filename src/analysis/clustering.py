# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from analysis_tools import *

# 1. Defining variables----------------------------------

# Defining the data file path
processed_data_path = "data/processed_data/"

# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = processed_data_path + "processed_destress_data_af2.csv"

# Defining file paths for labels
labels_df_path = processed_data_path + "labels_af2.csv"

# Defining the file path for the PCA model
pca_model_path = "analysis/dim_red/pca/pca_model"

# Defining file path for outliers
pca_outliers_path = "analysis/outlier_analysis/isolation_forest/both/pca_data_iso_for_labels.csv"

# Defining the list of number of clusters
kmeans_clusters_list = [2, 10, 20, 30, 40, 50]

# Output path
clustering_output_path = "analysis/clustering/"

# 2. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data = pd.read_csv(processed_destress_data_path)

# Reading in labels
labels_df = pd.read_csv(labels_df_path)

# Extracting features
features = processed_destress_data.columns.to_list()

# 3. Kmeans for different clusters----------------------------

print(processed_destress_data.columns.to_list())

wcss = []

processed_destress_data_clusters = processed_destress_data

for kmeans_clusters in kmeans_clusters_list:

    print(kmeans_clusters)

    # Fitting the model
    model = KMeans(n_clusters=kmeans_clusters, random_state=42)
    model_fit = model.fit(processed_destress_data)

    wcss.append(model.inertia_)

    # Extracting the labels
    predicted_labels = model_fit.labels_

    # Appending onto datasets
    processed_destress_data_clusters["kmeans_nclusters_" + str(kmeans_clusters)] = predicted_labels

# DBSCAN
model = DBSCAN()
model_fit = model.fit(processed_destress_data)

processed_destress_data_clusters["dbscan_default"] = model_fit.labels_



print(processed_destress_data_clusters)

processed_destress_data_labels = pd.concat([processed_destress_data_clusters, labels_df[["dssp_bin", "organism_scientific_name", "organism_group"]]], axis=1)

processed_destress_data_labels.to_csv(clustering_output_path + "processed_destress_data_clusters.csv", index=False)

# 4. Plotting-------------------------------------------------------------------

sns.lineplot(x=kmeans_clusters_list, y=wcss)
plt.savefig(
    clustering_output_path + "elbow_method.png",
    bbox_inches="tight",
    dpi=300,
)
plt.close()

# for kmeans_clusters in kmeans_clusters_list:

#     cmap=sns.color_palette("viridis", kmeans_clusters)

#     plot_latent_space_2d(data=pca_transformed_data_filt, 
#                         x="dim0", 
#                         y="dim1",
#                         axes_prefix = "PCA Dim",
#                         legend_title="clusters",
#                         hue="kmeans_nclusters_" + str(kmeans_clusters),
#                         # style=var,
#                         alpha=0.8, 
#                         s=20, 
#                         palette=cmap,
#                         output_path=clustering_output_path, 
#                         file_name="pca_embedding_robust_kmeans_nclusters" + str(kmeans_clusters))


for feature in (features + ["dssp_bin", "organism_group"]):
    print(feature)

    data = processed_destress_data_labels[[feature, "kmeans_nclusters_50"]].reset_index(drop=True)

    if feature in ["dssp_bin", "organism_group"]:

        sns.histplot(data=data, 
                     stat="count", 
                     multiple="stack",
                     x="kmeans_nclusters_50", 
                     kde=False,
                     hue=feature,
                     element="bars", 
                     legend=True)
        
    else: 

        sns.barplot(data=data,
                    x="kmeans_nclusters_50",
                    y=feature,
                    errorbar='sd')

    plt.savefig(
    clustering_output_path + "kmeans_nclusters50_" + feature + ".png",
    bbox_inches="tight",
    dpi=300,)
    plt.close()
    







