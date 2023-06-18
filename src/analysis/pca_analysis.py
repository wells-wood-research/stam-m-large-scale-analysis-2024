# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import os
from analysis_tools import *
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram

# 1. Defining variables----------------------------------

print(os.getcwd())

scaling_method = "robust"
pdb_or_af2 = "pdb"

# Defining the data file path
processed_data_path = "data/processed_data/" + pdb_or_af2 + "/" + scaling_method + "/"

# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = processed_data_path + "processed_destress_data.csv"

# Defining file paths for labels
labels_df_path = processed_data_path + "labels.csv"

# Defining the output paths
pca_analysis_path = "analysis/dim_red/pca/" + pdb_or_af2 + "/" + scaling_method + "/"

# Defining file path for outliers
pca_outliers_path = "analysis/outlier_analysis/isolation_forest/PDB/iso_for_outliers_robust0.05.csv"

# Setting random seed
np.random.seed(42)

# Defining number of components for PCA var plot
pca_var_plot_components = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
pca_num_components = 20
pca_hover_data = ["design_name", "dim0", "dim1", "dim2"]

# 1. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data_filt = pd.read_csv(processed_destress_data_path)

# Reading in labels
labels_df_filt = pd.read_csv(labels_df_path)

# # Reading in outliers pca data
# pca_outliers = pd.read_csv(pca_outliers_path)
# pca_outliers = pca_outliers[pca_outliers["iso_for_pred"] == -1]["design_name"].reset_index(drop=True)

# # Filtering htese out of the data
# processed_destress_data_filt = processed_destress_data[~labels_df["design_name"].isin(pca_outliers)].reset_index(drop=True)
# labels_df_filt = labels_df[~labels_df["design_name"].isin(pca_outliers)].reset_index(drop=True)

# 3. Performing PCA and plotting ---------------------------------------------------

pca_var_explained(data=processed_destress_data_filt, 
                  n_components_list=pca_var_plot_components, 
                  file_name="pca_var_explained",
                  output_path=pca_analysis_path)


pca_transformed_data, pca_model = perform_pca(data=processed_destress_data_filt, 
                                              labels_df = labels_df_filt,
                                              n_components = pca_num_components, 
                                              output_path = pca_analysis_path,
                                              file_path = "pca_transformed_data",
                                              components_file_path="comp_contrib")

# filehandler = open(pca_analysis_path + "pca_model", 'wb') 
# pickle.dump(pca_model, filehandler)

sns.set_theme(style="darkgrid")

labels_formatted = ['Design Name', 'Secondary Structure', 'PDB or AF2', 'Charge', 'Isoelectric Point', 'Rosetta Total Score', 'Packing Density', 'Hydrophobic Fitness', 'Aggrescan3d Average Score', "Source Organism"]

for i in range(0, len(labels_df_filt.columns.to_list())):

    var = labels_df_filt.columns.to_list()[i]
    label = labels_formatted[i]

    if var != "design_name":

        if var in ["isoelectric_point", "charge", "rosetta_total", "packing_density", "hydrophobic_fitness", "aggrescan3d_avg_value", "organism_scientific_name"]:

            cmap = sns.color_palette("viridis", as_cmap=True)

        else:

            cmap= sns.color_palette("tab10")

        # var = "organism_scientific_name"
        # label = "Source Organism"
        
        plot_pca_plotly(pca_data=pca_transformed_data, 
                        x="dim0", 
                        y="dim1", 
                        color=var, 
                        hover_data=pca_hover_data, 
                        opacity=0.8, 
                        size=5, 
                        output_path=pca_analysis_path, 
                        file_name="pca_embedding_" + var + ".html")
        
        plot_latent_space_2d(data=pca_transformed_data, 
                    x="dim0", 
                    y="dim1",
                    axes_prefix = "PCA Dim",
                    legend_title=label,
                    hue=var,
                    # style=var,
                    alpha=0.8, 
                    s=20, 
                    palette=cmap,
                    output_path=pca_analysis_path, 
                    file_name="pca_embedding_" + var)
        
# # Changing format of data from wide to long
# pca_transformed_data_long = pca_transformed_data.melt(
#     id_vars=[
#         # "pdb_or_af2",
#         # "dssp_bin",
#         # "isoelectric_point",
#         "organism_scientific_name",
#     ],
#     value_vars=[
#         "dim0",
#         "dim1",
#         "dim2",
#         "dim3",
#         "dim4",
#         "dim5",
#         "dim6",
#         "dim7",
#         "dim8",
#         "dim9",
#         "dim10",
#         "dim11",
#         "dim12",
#         "dim13",
#         "dim14",
#         "dim15",
#         "dim16",
#         "dim17",
#         "dim18",
#         "dim19",
        
#     ],
#     var_name="dim_id",
#     value_name="dim_value",
# )

# # Extracting the id of the pca dimension
# pca_transformed_data_long["dim_id"] = pca_transformed_data_long[
#     "dim_id"
# ].str.replace("dim", "")


# # Avering PCA dim value by dim id and organism
# pca_transformed_data_avg = pca_transformed_data.groupby(["organism_scientific_name"], as_index=False)[[
#         "dim0",
#         "dim1",
#         "dim2",
#         "dim3",
#         "dim4",
#         "dim5",
#         "dim6",
#         "dim7",
#         "dim8",
#         "dim9",
#         "dim10",
#         "dim11",
#         "dim12",
#         "dim13",
#         "dim14",
#         "dim15",
#         "dim16",
#         "dim17",
#         "dim18",
#         "dim19",
        
#     ]].apply(np.mean)

# pca_transformed_data_avg = pca_transformed_data_avg[pca_transformed_data_avg["organism_scientific_name"] != "Unknown"].reset_index(drop=True)

# distances_20d_all = distance_to_reference(data=pca_transformed_data_avg, 
#                       dim_columns=[
#                                 "dim0",
#                                 "dim1",
#                                 "dim2",
#                                 "dim3",
#                                 "dim4",
#                                 "dim5",
#                                 "dim6",
#                                 "dim7",
#                                 "dim8",
#                                 "dim9",
#                                 "dim10",
#                                 "dim11",
#                                 "dim12",
#                                 "dim13",
#                                 "dim14",
#                                 "dim15",
#                                 "dim16",
#                                 "dim17",
#                                 "dim18",
#                                 "dim19"], 
#                     distance_metric="euclidean", 
#                     output_path=pca_analysis_path)


# # distances_20d_all = squareform(distances_20d_all).reshape(-1, 1)

# # print(distances_20d_all)

# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)


# # setting distance_threshold=0 ensures we compute the full tree.
# model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='single', compute_distances=True)

# model_fitted = model.fit(distances_20d_all)


# plt.figure(figsize=(18, 10))
# # plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(model_fitted, truncate_mode=None, labels=distances_20d_all.columns.tolist(), orientation='left')
# plt.savefig(
#     pca_analysis_path + "pca_hierarchical_clustering_dendogram.png",
#     bbox_inches="tight",
#     dpi=600,
# )


# # labels=pca_transformed_data_avg.columns.to_list()



# # def convert_distance_matrix(distance_matrix):
# #     num_rows = len(distance_matrix)
# #     lower_triangular_matrix = []
    
# #     for i in range(1, num_rows + 1, 1):
# #         row = distance_matrix[i][:i]  # Extract the lower triangular elements
# #         lower_triangular_matrix.append(row)
    
# #     return lower_triangular_matrix


# # lower_triangular_matrix = convert_distance_matrix(distances_20d_all.values.tolist())

# # organism_names = distances_20d_all.columns.to_list()

# # del organism_names[0]

# # # Creating phlogenetic tree based on this distance matrix
# # phylo_tree = phylo_tree_from_dist_matrix(distance_matrix = lower_triangular_matrix, organism_names=organism_names)


# # print(phylo_tree)

# # pca_transformed_data_long_pdb_af2 = pca_transformed_data_long.groupby(["pdb_or_af2", "dim_id"], as_index=False)["dim_value"].agg([np.mean, np.std])
# # pca_transformed_data_long_pdb_af2.reset_index(inplace=True)

# # print(pca_transformed_data_long_pdb_af2)
        
        
# # spectral_plot(data=pca_transformed_data_long, 
# #                 x="dim_id", 
# #                 y="dim_value",
# #                 metric="pdb_or_af2", 
# #                 output_path=pca_analysis_path)

# # spectral_plot(data=pca_transformed_data_long[~pca_transformed_data_long["dssp_bin"].isin(["Mainly hbond turn", "Mainly bend", "Mainly 3 10 helix"])].reset_index(drop=True), 
# #                 x="dim_id", 
# #                 y="dim_value",
# #                 metric="dssp_bin", 
# #                 output_path=pca_analysis_path)

# # spectral_plot(data=pca_transformed_data_long, 
# #                 x="dim_id", 
# #                 y="dim_value",
# #                 metric="isoelectric_point", 
# #                 output_path=pca_analysis_path)


# # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))

# # # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))

# # organism_list = ["Arabidopsis thaliana",
# #                 "Glycine max",
# #                 "Oryza sativa",
# #                 "Zea mays"]

# # axes_list = [ax1, ax2, ax3, ax4]


# # for i in range(0, len(axes_list)):

# #     spectral_plot(data=pca_transformed_data_long[pca_transformed_data_long["organism_scientific_name"].isin([organism_list[i]])], 
# #                 x="dim_id", 
# #                 y="dim_value",
# #                 ax=axes_list[i],
# #                 metric="organism_scientific_name", 
# #                 output_path=pca_analysis_path)
    

# # plt.savefig(
# #     pca_analysis_path + "pca_spectral_comparison_af2_organism_plant.png",
# #     bbox_inches="tight",
# #     dpi=600,
# # )


# # spectral_plot(data=pca_transformed_data_long[~pca_transformed_data_long["organism_group"].isin(["Other"])], 
# #                 x="dim_id", 
# #                 y="dim_value",
# #                 metric="organism_group", 
# #                 output_path=pca_analysis_path,
# #                 file_name="pca_spectral_analysis_af2_organism_group")

