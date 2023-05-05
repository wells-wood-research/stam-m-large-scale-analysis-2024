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
destress_data_pdb_path = "data/processed_data/processed_destress_data_pdb.csv"

# Defining the file path for the labels
labels_pdb_file_path = "data/processed_data/labels_pdb.csv"

# Defining output path for outlier analysis
outlier_analysis_output = "analysis/outlier_analysis/"
pdb_isolation_forest_output = outlier_analysis_output + "isolation_forest/PDB/"

# PCA Parameters
pca_num_components=10

# Isolation forest parameters
iso_for_contamination = 0.01


# 2. Reading in data-----------------------------------------------------------------------

features_pdb = pd.read_csv(destress_data_pdb_path)
features_pdb_list = features_pdb.columns.to_list()
print(features_pdb_list)

labels_pdb = pd.read_csv(labels_pdb_file_path)

features_labels_pdb = pd.concat([features_pdb, labels_pdb[["design_name", "pdb_or_af2", "dssp_bin"]]], axis=1)


# 3. Outlier analysis------------------------------------------------------------------------

# All DE-STRESS features
iso_for = IsolationForest(random_state=42, contamination=iso_for_contamination).fit(features_pdb)
iso_for_pred = iso_for.predict(features_pdb)
iso_for_pred_df = pd.DataFrame(iso_for_pred, columns=["iso_for_pred"])
destress_data_iso_for = pd.concat([features_labels_pdb, iso_for_pred_df], axis=1)

destress_data_iso_for.to_csv(pdb_isolation_forest_output + "processed_destress_data_iso_for.csv", index=False)

sns.set_theme(style="darkgrid")

# for column in processed_destress_data_features.columns.to_list() + ["pdb_or_af2", "dssp_bin"]:
#     print(column)

#     sns.histplot(data=processed_destress_data_outliers, x=column, hue="iso_for_pred", element="step", stat="density", cumulative=True, fill=False, common_norm=False, palette=sns.color_palette("pastel"))

#     plt.savefig(pdb_isolation_forest_output + "iso_for_outlier_hist_" + column + ".png")
#     plt.close()


destress_data_outliers_list = destress_data_iso_for[destress_data_iso_for["iso_for_pred"] == -1].reset_index(drop=True)
destress_data_outliers_list.to_csv(pdb_isolation_forest_output + "pdb_iso_for_outliers_" + str(iso_for_contamination) + ".csv", index=False)

destress_data_outliers_removed = destress_data_iso_for[destress_data_iso_for["iso_for_pred"] != -1].reset_index(drop=True)
print(destress_data_outliers_removed)

destress_data_outliers_removed_features = destress_data_outliers_removed[features_pdb_list]

# 4. Performing PCA-------------------------------------------------------------------------

pca_transformed_data_with_outliers, pca_model_with_outliers = perform_pca(data=features_pdb, 
                                              labels_df = iso_for_pred_df,
                                              n_components = pca_num_components, 
                                              output_path = pdb_isolation_forest_output,
                                              file_path = "pca_transformed_data_with_outliers",
                                              components_file_path="pdb_with_outliers")
cmap= sns.color_palette("tab10")

plot_latent_space_2d(data=pca_transformed_data_with_outliers, 
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

pca_transformed_outliers_removed = pca_model_with_outliers.transform(destress_data_outliers_removed_features)
# Converting to data frame and renaming columns
pca_transformed_outliers_removed = pd.DataFrame(pca_transformed_outliers_removed).rename(
    columns={0: "dim0", 1: "dim1", 2: "dim2", 3: "dim3", 4: "dim4", 5: "dim5", 6: "dim6", 7: "dim7", 8: "dim8", 9: "dim9"}
)
pca_transformed_outliers_removed_labels = pd.concat([pca_transformed_outliers_removed, destress_data_outliers_removed["iso_for_pred"]], axis=1)

plot_latent_space_2d(data=pca_transformed_outliers_removed_labels, 
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
                    file_name="pca_embedding_without_outliers_same_pca")


pca_transformed_data_without_outliers, pca_model_without_outliers = perform_pca(data=destress_data_outliers_removed_features, 
                                              labels_df = destress_data_outliers_removed["iso_for_pred"],
                                              n_components = pca_num_components, 
                                              output_path = pdb_isolation_forest_output,
                                              file_path = "pca_transformed_data_without_outliers",
                                              components_file_path="pdb_without_outliers")
cmap= sns.color_palette("tab10")

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