# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from analysis_tools import *

# 1. Defining variables----------------------------------

# Defining the data file path
processed_data_path = "data/processed_data/"

# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = processed_data_path + "processed_destress_data.csv"

# Defining file paths for labels
labels_df_path = processed_data_path + "labels.csv"

# Defining the file path for the PCA model
pca_model_path = "analysis/dim_red/pca/pca_model"

# Defining file path for outliers
pca_outliers_path = "analysis/outlier_analysis/isolation_forest/both/pca_data_iso_for_labels.csv"

# Defining the list of number of clusters
kmeans_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Output path
clustering_output_path = "analysis/clustering/"

# 2. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data = pd.read_csv(processed_destress_data_path)

# Reading in labels
labels_df = pd.read_csv(labels_df_path)

# Reading in pca model
file = open(pca_model_path, 'rb')
pca_model = pickle.load(file)

# Reading in outliers pca data
pca_outliers = pd.read_csv(pca_outliers_path)
pca_outliers = pca_outliers[pca_outliers["iso_for_pred"] == -1]["design_name"].reset_index(drop=True)

# Filtering htese out of the data
processed_destress_data_filt = processed_destress_data[~labels_df["design_name"].isin(pca_outliers)].reset_index(drop=True)
labels_df_filt = labels_df[~labels_df["design_name"].isin(pca_outliers)].reset_index(drop=True)

# PCA transformed data
pca_transformed_data_filt = pca_model.transform(processed_destress_data_filt)
pca_transformed_data_filt = pd.DataFrame(pca_transformed_data_filt).rename(
        columns={0: "dim0", 1: "dim1", 2: "dim2", 3: "dim3", 4: "dim4", 5: "dim5", 6: "dim6", 7: "dim7", 8: "dim8", 9: "dim9"}
    )
pca_transformed_data_filt = pd.concat([pca_transformed_data_filt, labels_df_filt], axis=1)

# 3. Kmeans for different clusters----------------------------

print(processed_destress_data_filt.columns.to_list())

wcss = []

for kmeans_clusters in kmeans_clusters_list:

    print(kmeans_clusters)

    # Fitting the model
    model = KMeans(n_clusters=kmeans_clusters, random_state=42)
    model_fit = model.fit(processed_destress_data_filt)

    wcss.append(model.inertia_)

    # Extracting the labels
    predicted_labels = model_fit.labels_

    # Appending onto datasets
    processed_destress_data_filt["kmeans_nclusters_" + str(kmeans_clusters)] = predicted_labels
    pca_transformed_data_filt["kmeans_nclusters_" + str(kmeans_clusters)] = predicted_labels


print(processed_destress_data_filt)
print(pca_transformed_data_filt)

processed_destress_data_filt.to_csv(clustering_output_path + "processed_destress_data_clusters.csv")
pca_transformed_data_filt.to_csv(clustering_output_path + "pca_transformed_data_clusters.csv")

# 4. Plotting-------------------------------------------------------------------

sns.lineplot(x=kmeans_clusters_list, y=wcss)
plt.savefig(
    clustering_output_path + "elbow_method.png",
    bbox_inches="tight",
    dpi=300,
)
plt.close()

for kmeans_clusters in kmeans_clusters_list:

    cmap=sns.color_palette("viridis", kmeans_clusters)

    plot_latent_space_2d(data=pca_transformed_data_filt, 
                        x="dim0", 
                        y="dim1",
                        axes_prefix = "PCA Dim",
                        legend_title="clusters",
                        hue="kmeans_nclusters_" + str(kmeans_clusters),
                        # style=var,
                        alpha=0.8, 
                        s=20, 
                        palette=cmap,
                        output_path=clustering_output_path, 
                        file_name="pca_embedding_robust_kmeans_nclusters" + str(kmeans_clusters))








