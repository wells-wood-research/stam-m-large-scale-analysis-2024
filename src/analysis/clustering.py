# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

# 1. Defining variables----------------------------------

# Defining the data file path
processed_data_path = "data/processed_data/"

# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = processed_data_path + "processed_destress_data_pdb.csv"

# Defining file paths for labels
labels_df_path = processed_data_path + "labels_pdb.csv"

# Defining the list of number of clusters
# kmeans_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
kmeans_clusters_list = [2]

# 2. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data = pd.read_csv(processed_destress_data_path)

# Reading in labels
labels_df = pd.read_csv(labels_df_path)

# 3. Kmeans for different clusters----------------------------

print(processed_destress_data.columns.to_list())

model_list = []
pred_list = []
sil_score_list = []

# Calculate pairwise distances for the processed DE-STRESS data
euclidean_pairwise_distances = pairwise_distances(X=processed_destress_data,
                                                  Y=processed_destress_data, 
                                                  metric='euclidean')

for kmeans_clusters in kmeans_clusters_list:

    print(kmeans_clusters)

    # Fitting the model
    model = KMeans(n_clusters=kmeans_clusters, random_state=42)
    model_fit = model.fit(processed_destress_data)

    # Extracting the labels
    predicted_labels = model_fit.labels_

    # Calculating the silhouette score
    sil_score = silhouette_score(X=euclidean_pairwise_distances, labels=predicted_labels, metric="precomputed")

    # Appending to lists
    model_list.append(model)
    pred_list.append(predicted_labels)
    sil_score_list.append(sil_score)


print(model_list)
print(pred_list)
print(sil_score_list)

# Plotting silhouette score by number of clusters
sns.scatterplot(x=kmeans_clusters_list,y=sil_score_list)
plt.show()





