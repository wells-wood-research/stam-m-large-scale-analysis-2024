# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import os
from analysis_tools import *
import pickle

# 1. Defining variables----------------------------------

print(os.getcwd())

# Defining the data file path
processed_data_path = "data/processed_data/"

# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = processed_data_path + "processed_destress_data.csv"

# Defining file paths for labels
labels_df_path = processed_data_path + "labels.csv"

# Defining the output paths
pca_analysis_path = "analysis/dim_red/pca/"

# Defining file path for outliers
pca_outliers_path = "analysis/outlier_analysis/isolation_forest/both/pca_data_iso_for_labels.csv"

# Setting random seed
np.random.seed(42)

# Defining number of components for PCA var plot
pca_var_plot_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]
pca_num_components = 10
pca_hover_data = ["design_name", "dim0", "dim1", "dim2"]

# 1. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data = pd.read_csv(processed_destress_data_path)

# Reading in labels
labels_df = pd.read_csv(labels_df_path)

# Reading in outliers pca data
pca_outliers = pd.read_csv(pca_outliers_path)
pca_outliers = pca_outliers[pca_outliers["iso_for_pred"] == -1]["design_name"].reset_index(drop=True)

# Filtering htese out of the data
processed_destress_data_filt = processed_destress_data[~labels_df["design_name"].isin(pca_outliers)].reset_index(drop=True)
labels_df_filt = labels_df[~labels_df["design_name"].isin(pca_outliers)].reset_index(drop=True)

# 3. Performing PCA and plotting ---------------------------------------------------

pca_var_explained(data=processed_destress_data_filt, 
                  n_components_list=pca_var_plot_components, 
                  output_path=pca_analysis_path)


pca_transformed_data, pca_model = perform_pca(data=processed_destress_data_filt, 
                                              labels_df = labels_df_filt,
                                              n_components = pca_num_components, 
                                              output_path = pca_analysis_path,
                                              file_path = "pca_transformed_data_robust",
                                              components_file_path="robust")

filehandler = open(pca_analysis_path + "pca_model", 'wb') 
pickle.dump(pca_model, filehandler)

sns.set_theme(style="darkgrid")

labels_formatted = ['Design Name', 'Secondary Structure', 'PDB or AF2', 'Charge', 'Isoelectric Point', 'Rosetta Total Score', 'Packing Density', 'Hydrophobic Fitness', 'Aggrescan3d Average Score']

for i in range(0, len(labels_df.columns.to_list())):

    var = labels_df.columns.to_list()[i]
    label = labels_formatted[i]

    if var != "design_name":

        if var in ["isoelectric_point", "charge", "rosetta_total", "packing_density", "hydrophobic_fitness", "aggrescan3d_avg_value"]:

            cmap = sns.color_palette("viridis", as_cmap=True)

        else:

            cmap= sns.color_palette("tab10")
        
        plot_pca_plotly(pca_data=pca_transformed_data, 
                        x="dim0", 
                        y="dim1", 
                        color=var, 
                        hover_data=pca_hover_data, 
                        opacity=0.8, 
                        size=5, 
                        output_path=pca_analysis_path, 
                        file_name="pca_embedding_robust_" + var + ".html")
        
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
                    file_name="pca_embedding_robust_" + var)
        
# Changing format of data from wide to long
pca_transformed_data_long = pca_transformed_data.melt(
    id_vars=[
        "pdb_or_af2",
        "dssp_bin",
        "isoelectric_point",
    ],
    value_vars=[
        "dim0",
        "dim1",
        "dim2",
        "dim3",
        "dim4",
        "dim5",
        "dim6",
        "dim7",
        "dim8",
        "dim9",
    ],
    var_name="dim_id",
    value_name="dim_value",
)

# Extracting the id of the pca dimension
pca_transformed_data_long["dim_id"] = pca_transformed_data_long[
    "dim_id"
].str.replace("dim", "")


print(pca_transformed_data_long)
print(pca_transformed_data_long.columns.to_list())


# pca_transformed_data_long_pdb_af2 = pca_transformed_data_long.groupby(["pdb_or_af2", "dim_id"], as_index=False)["dim_value"].agg([np.mean, np.std])
# pca_transformed_data_long_pdb_af2.reset_index(inplace=True)

# print(pca_transformed_data_long_pdb_af2)
        
        
spectral_plot(data=pca_transformed_data_long, 
                x="dim_id", 
                y="dim_value",
                metric="pdb_or_af2", 
                output_path=pca_analysis_path)

spectral_plot(data=pca_transformed_data_long[~pca_transformed_data_long["dssp_bin"].isin(["Mainly hbond turn", "Mainly bend", "Mainly 3 10 helix"])].reset_index(drop=True), 
                x="dim_id", 
                y="dim_value",
                metric="dssp_bin", 
                output_path=pca_analysis_path)

spectral_plot(data=pca_transformed_data_long, 
                x="dim_id", 
                y="dim_value",
                metric="isoelectric_point", 
                output_path=pca_analysis_path)

