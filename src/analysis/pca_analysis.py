# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import os
from analysis_tools import *



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

# 3. Performing PCA and plotting ---------------------------------------------------

pca_var_explained(data=processed_destress_data, 
                  n_components_list=pca_var_plot_components, 
                  output_path=pca_analysis_path)


pca_transformed_data = perform_pca(data=processed_destress_data, 
                                   labels_df = labels_df,
                                   n_components = pca_num_components, 
                                   output_path = pca_analysis_path)

for var in labels_df.columns.to_list():
    if var != "design_name":

        # plot_pca_plotly(pca_data=pca_transformed_data, 
        #                 x="dim0", 
        #                 y="dim1", 
        #                 color=var, 
        #                 hover_data=pca_hover_data, 
        #                 opacity=0.6, 
        #                 size=20, 
        #                 output_path=pca_analysis_path, 
        #                 file_name="pca_embedding_" + var + ".html")
        
        plot_pca_2d(pca_data=pca_transformed_data[pca_transformed_data["pdb_or_af2"] == "AF2"].reset_index(drop=True), 
                    x="dim0", 
                    y="dim1", 
                    hue=var,
                    # style=var,
                    alpha=0.8, 
                    s=40, 
                    output_path=pca_analysis_path, 
                    file_name="pca_embedding_" + var)

# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(
#     6, 2, figsize=(4, 6), sharey=True, sharex=True
# )
# fig.suptitle("Histograms of the principal components for all PDB and AF2 structures")

# ax1.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "PDB"]["pca_dim0"],
#     bins=20,
#     alpha=0.5,
#     label="PDB",
#     density=True,
#     histtype="stepfilled",
#     color="tab:blue",
# )
# # ax1.set_xlabel("PC1")
# # ax1.set_ylabel("Density")
# # ax1.legend(loc="upper right")

# ax2.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "PDB"]["pca_dim1"],
#     bins=20,
#     alpha=0.5,
#     label="PDB",
#     density=True,
#     histtype="stepfilled",
#     color="tab:blue",
# )
# # ax2.set_xlabel("PC2")
# # ax2.set_ylabel("Density")
# ax2.legend(loc="upper right")
# sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1), frameon=False)

# ax3.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] != "PDB"]["pca_dim0"],
#     bins=20,
#     alpha=0.5,
#     label="AF2",
#     density=True,
#     histtype="stepfilled",
#     color="tab:orange",
# )
# # ax3.set_xlabel("PC1")
# # ax3.set_ylabel("Density")
# # ax3.legend(loc="upper right")


# ax4.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] != "PDB"]["pca_dim1"],
#     bins=20,
#     alpha=0.5,
#     label="AF2",
#     density=True,
#     histtype="stepfilled",
#     color="tab:orange",
# )
# # ax4.set_xlabel("PC2")
# # ax4.set_ylabel("Density")
# ax4.legend(loc="upper right")
# sns.move_legend(ax4, "upper left", bbox_to_anchor=(1, 1), frameon=False)

# ax5.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mainly loop"]["pca_dim0"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mainly Loop",
#     density=True,
#     histtype="stepfilled",
#     color="tab:green",
# )
# # ax5.set_xlabel("PC1")
# # ax5.set_ylabel("Density")
# # ax5.legend(loc="upper right")


# ax6.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mainly loop"]["pca_dim1"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mainly Loop",
#     density=True,
#     histtype="stepfilled",
#     color="tab:green",
# )
# # ax6.set_xlabel("PC2")
# # ax6.set_ylabel("Density")
# ax6.legend(loc="upper right")
# sns.move_legend(ax6, "upper left", bbox_to_anchor=(1, 1), frameon=False)

# ax7.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mainly alpha helix"]["pca_dim0"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mainly Alpha Helix",
#     density=True,
#     histtype="stepfilled",
#     color="tab:purple",
# )
# # ax7.set_xlabel("PC1")
# # ax7.set_ylabel("Density")
# # ax7.legend(loc="upper right")


# ax8.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mainly alpha helix"]["pca_dim1"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mainly Alpha Helix",
#     density=True,
#     histtype="stepfilled",
#     color="tab:purple",
# )
# # ax8.set_xlabel("PC2")
# # ax8.set_ylabel("Density")
# ax8.legend(loc="upper right")
# sns.move_legend(ax8, "upper left", bbox_to_anchor=(1, 1), frameon=False)

# ax9.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mixed"]["pca_dim0"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mixed",
#     density=True,
#     histtype="stepfilled",
#     color="tab:red",
# )
# # ax9.set_xlabel("PC1")
# # ax9.set_ylabel("Density")
# # ax9.legend(loc="upper right")


# ax10.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mixed"]["pca_dim1"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mixed",
#     density=True,
#     histtype="stepfilled",
#     color="tab:red",
# )
# # ax10.set_xlabel("PC2")
# # ax10.set_ylabel("Density")
# ax10.legend(loc="upper right")
# sns.move_legend(ax10, "upper left", bbox_to_anchor=(1, 1), frameon=False)


# ax11.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mainly beta strand"]["pca_dim0"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mixed",
#     density=True,
#     histtype="stepfilled",
#     color="tab:cyan",
# )
# ax11.set_xlabel("PC1")
# # ax11.set_ylabel("Density")
# # ax11.legend(loc="upper right")


# ax12.hist(
#     pca_transformed_df[pca_transformed_df["dssp_bin"] == "Mainly beta strand"]["pca_dim1"],
#     bins=20,
#     alpha=0.5,
#     label="AF2 Mainly Beta Strand",
#     density=True,
#     histtype="stepfilled",
#     color="tab:cyan",
# )
# ax12.set_xlabel("PC2")
# # ax12.set_ylabel("Density")
# ax12.legend(loc="upper right")
# sns.move_legend(ax12, "upper left", bbox_to_anchor=(1, 1), frameon=False)

# plt.savefig(
#     dim_red_analysis_path + "pca_hists_pdb_mainlyloop.png",
#     bbox_inches="tight",
#     dpi=600,
# )
# plt.close()



# # Plotting the gplvm with plotly and overlaying different variables
# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="pdb_or_af2",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_pdb_or_af2.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="dssp_bin",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_dssp_bin.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="isoelectric_point",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_isoelectric_point.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="packing_density",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_packing_density.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="hydrophobic_fitness",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_hydrophobic_fitness.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="aggrescan3d_avg_value",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_aggrescan3d_avg_value.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="rosetta_total",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_rosetta_total.html")

# fig = px.scatter(
#     X,
#     x="gplvm_dim_0",
#     y="gplvm_dim_1",
#     color="organism_scientific_name",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html(dim_red_analysis_path + "gplvm_organism.html")


