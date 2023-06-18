# 0. Importing packages---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pickle
from analysis_tools import *


# 1. Defining variables----------------------------------------------------------------

# Defining a file path for the processed DE-STRESS data
destress_data_path = "data/processed_data/processed_destress_data_pdb.csv"

# Defining the file path for the labels
labels_file_path = "data/processed_data/labels_pdb.csv"

# Defining output path for outlier analysis
outlier_analysis_output = "analysis/outlier_analysis/"
isolation_forest_output = outlier_analysis_output + "isolation_forest/PDB/"
local_outlier_output = outlier_analysis_output + "local_outlier/PDB/"

# PCA Parameters
pca_num_components=6

# Isolation forest parameters
iso_for_contamination = 0.02

# Setting seed
np.random.seed(42)


# 2. Reading in data-----------------------------------------------------------------------

features = pd.read_csv(destress_data_path)
features_list = features.columns.to_list()
print(features_list)

labels = pd.read_csv(labels_file_path)

features_labels = pd.concat([features, labels[["design_name", "pdb_or_af2", "dssp_bin"]]], axis=1)

# 3. Performing PCA-------------------------------------------------------------------------

pca_var_explained(data=features, n_components_list=range(2, pca_num_components, 1), file_name = "pca_var_explained_with_outliers_robust", output_path=isolation_forest_output)

pca_transformed_data, pca_model_with_outliers = perform_pca(data=features, 
                                                            labels_df = None,
                                                            n_components = pca_num_components, 
                                                            output_path = isolation_forest_output,
                                                            file_path = "pca_transformed_data_with_outliers_robust",
                                                            components_file_path="with_outliers_robust")


pca_transformed_data_labels = pd.concat([pca_transformed_data, labels[["design_name", "pdb_or_af2", "dssp_bin"]]], axis=1)


sns.set_theme(style="darkgrid")

cmap= sns.color_palette("tab10")

plot_latent_space_2d(data=pca_transformed_data_labels, 
                    x="dim0", 
                    y="dim1",
                    axes_prefix = "PCA Dim",
                    legend_title="label",
                    hue="pdb_or_af2",
                    # style=var,
                    alpha=0.9, 
                    s=4, 
                    palette=cmap,
                    output_path=isolation_forest_output, 
                    file_name="pca_embedding_with_outliers_robust_")

# 3. Outlier analysis------------------------------------------------------------------------

# Isolation Forest

iso_for_outliers = outlier_detection_iso_for(data=pca_transformed_data, 
                                            labels=labels, 
                                            contamination=iso_for_contamination, 
                                            n_estimators=10000, 
                                            max_features=pca_num_components, 
                                            output_path=isolation_forest_output, 
                                            file_name="iso_for_outliers_robust")

features_outliers_removed = features[iso_for_outliers["iso_for_pred"] != -1].reset_index(drop=True)
labels_outliers_removed = labels[iso_for_outliers["iso_for_pred"] != -1].reset_index(drop=True)

pca_var_explained(data=features_outliers_removed, n_components_list=range(2, pca_num_components, 1), file_name = "pca_var_explained_without_outliers", output_path=isolation_forest_output)

pca_transformed_data_without_outliers, pca_model_without_outliers = perform_pca(data=features_outliers_removed, 
                                              labels_df = labels_outliers_removed,
                                              n_components = pca_num_components, 
                                              output_path = isolation_forest_output,
                                              file_path = "pca_transformed_data_without_outliers_robust",
                                              components_file_path="without_outliers_robust")


plot_latent_space_2d(data=pca_transformed_data_without_outliers, 
                    x="dim0", 
                    y="dim1",
                    axes_prefix = "PCA Dim",
                    legend_title="label",
                    hue="pdb_or_af2",
                    # style=var,
                    alpha=0.9, 
                    s=4, 
                    palette=cmap,
                    output_path=isolation_forest_output, 
                    file_name="pca_embedding_without_outliers_different_pca_robust")



