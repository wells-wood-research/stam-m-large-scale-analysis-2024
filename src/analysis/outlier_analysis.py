# 0. Importing packages---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
import pickle
from analysis_tools import *


# 1. Defining variables----------------------------------------------------------------

# Defining a file path for the processed DE-STRESS data
destress_data_path = "data/processed_data/processed_destress_data.csv"

# Defining the file path for the labels
labels_file_path = "data/processed_data/labels.csv"

# Defining output path for outlier analysis
outlier_analysis_output = "analysis/outlier_analysis/"
pdb_isolation_forest_output = outlier_analysis_output + "isolation_forest/both/"

# PCA Parameters
pca_num_components=2

# Isolation forest parameters
iso_for_contamination = 0.01


# 2. Reading in data-----------------------------------------------------------------------

features = pd.read_csv(destress_data_path)
features_list = features.columns.to_list()
print(features_list)

labels = pd.read_csv(labels_file_path)

features_labels = pd.concat([features, labels[["design_name", "pdb_or_af2", "dssp_bin"]]], axis=1)


# 3. Performing PCA-------------------------------------------------------------------------

pca_transformed_data, pca_model_with_outliers = perform_pca(data=features, 
                                                            labels_df = None,
                                                            n_components = pca_num_components, 
                                                            output_path = pdb_isolation_forest_output,
                                                            file_path = "pca_transformed_data_with_outliers",
                                                            components_file_path="with_outliers")


pca_transformed_data_labels = pd.concat([pca_transformed_data, labels[["design_name", "pdb_or_af2", "dssp_bin"]]], axis=1)


# 4. Outlier analysis------------------------------------------------------------------------

# All DE-STRESS features
iso_for = IsolationForest(random_state=42, contamination=iso_for_contamination).fit(pca_transformed_data)
iso_for_pred = iso_for.predict(pca_transformed_data)
iso_for_pred_df = pd.DataFrame(iso_for_pred, columns=["iso_for_pred"])
pca_data_iso_for = pd.concat([pca_transformed_data, iso_for_pred_df], axis=1)
pca_data_iso_for_labels = pd.concat([pca_transformed_data_labels, iso_for_pred_df], axis=1)

pca_data_iso_for_labels.to_csv(pdb_isolation_forest_output + "pca_data_iso_for_labels.csv", index=False)

sns.set_theme(style="darkgrid")

# for column in processed_destress_data_features.columns.to_list() + ["pdb_or_af2", "dssp_bin"]:
#     print(column)

#     sns.histplot(data=processed_destress_data_outliers, x=column, hue="iso_for_pred", element="step", stat="density", cumulative=True, fill=False, common_norm=False, palette=sns.color_palette("pastel"))

#     plt.savefig(pdb_isolation_forest_output + "iso_for_outlier_hist_" + column + ".png")
#     plt.close()


pca_data_iso_for_outliers_list = pca_data_iso_for[pca_data_iso_for["iso_for_pred"] == -1].reset_index(drop=True)
pca_data_iso_for_outliers_list.to_csv(pdb_isolation_forest_output + "pdb_iso_for_outliers_" + str(iso_for_contamination) + ".csv", index=False)

destress_data_iso_for_outliers_removed = features_labels[pca_data_iso_for["iso_for_pred"] != -1].reset_index(drop=True)
print(destress_data_iso_for_outliers_removed)

pca_iso_for_outliers_removed = pca_data_iso_for[pca_data_iso_for["iso_for_pred"] != -1].reset_index(drop=True)
print(pca_iso_for_outliers_removed)

destress_data_outliers_removed_features = destress_data_iso_for_outliers_removed[features_list]

# 4. Performing PCA-------------------------------------------------------------------------

cmap= sns.color_palette("tab10")

plot_latent_space_2d(data=pca_data_iso_for, 
                    x="dim0", 
                    y="dim1",
                    axes_prefix = "PCA Dim",
                    legend_title="label",
                    hue="iso_for_pred",
                    # style=var,
                    alpha=0.9, 
                    s=4, 
                    palette=cmap,
                    output_path=pdb_isolation_forest_output, 
                    file_name="pca_embedding_with_outliers_")


pca_transformed_data_without_outliers, pca_model_without_outliers = perform_pca(data=destress_data_outliers_removed_features, 
                                              labels_df = pca_iso_for_outliers_removed["iso_for_pred"],
                                              n_components = pca_num_components, 
                                              output_path = pdb_isolation_forest_output,
                                              file_path = "pca_transformed_data_without_outliers",
                                              components_file_path="without_outliers")
cmap= sns.color_palette("tab10")

print(pca_transformed_data_without_outliers)

plot_latent_space_2d(data=pca_transformed_data_without_outliers, 
                    x="dim0", 
                    y="dim1",
                    axes_prefix = "PCA Dim",
                    legend_title="label",
                    hue="iso_for_pred",
                    # style=var,
                    alpha=0.9, 
                    s=4, 
                    palette=cmap,
                    output_path=pdb_isolation_forest_output, 
                    file_name="pca_embedding_without_outliers_different_pca")
