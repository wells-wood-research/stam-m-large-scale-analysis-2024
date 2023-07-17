# This script performs KMeans on the different PCA transformed data sets
# to find clusters and evaluating the clusters against organism groups

# 0. Importing packages and helper functions---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics

# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination_list = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining number of components for gaussian mixture model
gaussian_mixture_components = 4

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

            # Output path
            clustering_output_path = (
                "analysis/clustering/"
                + dataset
                + "/"
                + "iso_for_"
                + str(iso_for_contamination)
                + "/"
                + scaling_method
                + "/"
            )

            # Defining the path for processed DE-STRESS data
            processed_destress_data_path = (
                processed_data_path + "processed_destress_data_scaled.csv"
            )

            # Defining the path for the transformed PCA data
            pca_transformed_data_path = pca_analysis_path + "pca_transformed_data.csv"

            # Defining file paths for labels
            labels_df_path = processed_data_path + "labels.csv"

            # 3. Reading in data------------------------------------------------------------------------

            # Reading in processed destress data
            processed_destress_data = pd.read_csv(processed_destress_data_path)

            # Reading in pca transformed data
            pca_transformed_data = pd.read_csv(pca_transformed_data_path)

            # Only selecting pca dim columns
            pca_transformed_data_filt = pca_transformed_data[
                pca_transformed_data["organism_group"].isin(["Unknown", "Other"])
            ].reset_index(drop=True)
            # pca_transformed_data_dims = pca_transformed_data_filt.filter(regex="dim")

            # Only selecting pca dim columns
            pca_transformed_data_dims = pca_transformed_data_filt[["dim0", "dim1"]]

            # # Reading in labels
            # labels_df = pd.read_csv(labels_df_path)

            # 4. Gaussian Mixture Model------------------------------------------------------------------

            model = DBSCAN(eps=0.2, min_samples=50).fit(pca_transformed_data_dims)
            predicted_labels = model.labels_

            print(predicted_labels)
            print(np.unique(predicted_labels))
            # model = GaussianMixture(
            #     n_components=gaussian_mixture_components, random_state=42
            # ).fit(pca_transformed_data_dims)

            # predicted_labels = model.predict(pca_transformed_data_dims)

            # print(predicted_labels)

            # # Fitting the model
            # model = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init="auto")
            # model_fit = model.fit(pca_transformed_data_filt)

            # # # Fitting the model
            # # model = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init="auto")
            # # model_fit = model.fit(processed_destress_data)

            # # Extracting the labels
            # predicted_labels = model_fit.labels_

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

            # rand_score = metrics.rand_score(
            #     pca_transformed_data_filt["organism_group"].to_list(), predicted_labels
            # )

            adj_rand_score = metrics.adjusted_rand_score(
                pca_transformed_data_filt["organism_group"].to_list(), predicted_labels
            )

            print(adj_rand_score)
