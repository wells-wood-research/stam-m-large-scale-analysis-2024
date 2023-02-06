# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
# import umap
import plotly.graph_objs as go
# import GPy
import os

import torch
from torch.nn import Parameter
import pyro 
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(1)

# 1. Defining variables----------------------------------

print(os.getcwd())
# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = "data/processed_data/processed_destress_data.csv"

# Defining the output paths
dim_red_analysis_path = "analysis/dim_red/"

# Setting random seed
np.random.seed(42)

# 1. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data = pd.read_csv(processed_destress_data_path)

# 2. Processing data----------------------------------------

# Extracting columns as variables before scaling
design_name = processed_destress_data["design_name"]
# uniprot_id = processed_destress_data["uniprot_id"]
pdb_or_af2 = processed_destress_data["pdb_or_af2"]
dssp_bin = processed_destress_data["dssp_bin"]
isoelectric_point_bin = processed_destress_data["isoelectric_point"]
# organism_scientific_name = processed_destress_data["organism_scientific_name"]

# Dropping columns that are not needed anymore
processed_destress_data.drop(
    [
        "design_name",
        "file_name",
        "pdb_or_af2",
        "dssp_bin",
        # "organism_scientific_name",
        # "uniprot_id",
    ],
    axis=1,
    inplace=True,
)

print(processed_destress_data.columns.to_list())

# 3. Scaling data---------------------------------------------------------------

# Using the min max scaler to scale the data set to be between 0 and 1
scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(processed_destress_data)

# Joining the columns back on
scaled_df = pd.DataFrame(scaled_df, columns=processed_destress_data.columns)

scaled_df_arr = scaled_df.to_numpy()

# # 4. Performing PCA and plotting components
# # against variance with two different scaling methods---------------------------------------------------
# n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# # Creating a data frame to collect results
# var_explained_df = pd.DataFrame(columns=["n_components", "var_explained"])

# # Looping through different scaling methods and components
# for i in n_components:

#     # Performing PCA with the specified components
#     pca = decomposition.PCA(n_components=i)
#     pca.fit(scaled_df)

#     # Calculating the variance explained
#     var_explained = np.sum(pca.explained_variance_ratio_)

#     # Appending to the data frame
#     var_explained_df = var_explained_df.append(
#         {"n_components": i, "var_explained": var_explained},
#         ignore_index=True,
#     )

# # Saving as a csv file
# var_explained_df.to_csv(
#     dim_red_analysis_path + "var_explained.csv",
#     index=False,
# )

# sns.set_theme(style="darkgrid")

# # Plotting the data and saving
# var_plot = sns.lineplot(
#     x="n_components",
#     y="var_explained",
#     data=var_explained_df,
# )
# plt.title("""Variance explained by number of pca components and scaling method.""")
# plt.xlabel("Number of components")
# plt.legend(loc="upper right", ncol=2, handletextpad=0.1)
# plt.savefig(dim_red_analysis_path + "var_explained.png")
# plt.close()

# # Performing PCA with 2 components
# pca = decomposition.PCA(n_components=10)
# pca.fit(scaled_df)

# # Saving contributions of the features to the principal components
# feat_contr_to_cmpts = pd.DataFrame(
#     np.round(abs(pca.components_), 4), columns=scaled_df.columns
# )
# feat_contr_to_cmpts.to_csv(
#     dim_red_analysis_path + "feat_contr_to_cmpts.csv", index=True
# )

# # Selecting the 10 largest contributers to pca component 1
# components_1_contr = feat_contr_to_cmpts.iloc[0].nlargest(10, keep="first")
# components_1_contr.to_csv(dim_red_analysis_path + "components_1_contr.csv", index=True)

# # Selecting the 10 largest contributers to pca component 2
# components_2_contr = feat_contr_to_cmpts.iloc[1].nlargest(10, keep="first")
# components_2_contr.to_csv(dim_red_analysis_path + "components_2_contr.csv", index=True)

# # Transforming the data
# pca_transformed_df = pca.transform(scaled_df)

# # Converting to a data frame and renaming columns
# pca_transformed_df = pd.DataFrame(pca_transformed_df).rename(
#     columns={
#         0: "pca_dim0",
#         1: "pca_dim1",
#         2: "pca_dim2",
#         3: "pca_dim3",
#         4: "pca_dim4",
#         5: "pca_dim5",
#         6: "pca_dim6",
#         7: "pca_dim7",
#         8: "pca_dim8",
#         9: "pca_dim9",
#     }
# )

# # Adding the labels back
# pca_transformed_df = pd.concat(
#     [
#         design_name,
#         pdb_or_af2,
#         dssp_bin,
#         isoelectric_point_bin,
#         pca_transformed_df,
#         organism_scientific_name,
#         ],
#     axis=1,
# )

# # Outputting the transformed data
# pca_transformed_df.to_csv(
#     "data/pca_transformed_data.csv",
#     index=False,
# )

# # pca_transformed_df_test = pca_transformed_df[pca_transformed_df["pca_dim0"] > 0.6]

# # # Outputting the transformed data
# # pca_transformed_df_test.to_csv(
# #     "data/pca_transformed_df_test.csv",
# #     index=False,
# # )

# # print(pca_transformed_df)

# # 7. Plotting the PCA embedding---------------------------------------------

# fig = px.scatter(
#     pca_transformed_df,
#     x="pca_dim0",
#     y="pca_dim1",
#     # z="pca_dim2",
#     color="organism_scientific_name",
#     color_discrete_sequence=px.colors.qualitative.G10,
#     symbol="pdb_or_af2",
#     hover_data=["design_name", "pca_dim0", "pca_dim1", "pca_dim2"],
#     opacity=0.6,
# )
# fig.update_traces(
#     marker=dict(size=20, line=dict(width=2)),
#     selector=dict(mode="markers"),
# )
# fig.write_html("analysis/pdb_vs_af2_pca_embedding.html")
# fig.show()

# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(
#     6, 2, figsize=(4, 6), sharey=True, sharex=True
# )
# # fig.suptitle("Histograms of the principal components for all PDB and AF2 structures")

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


# 8. Fitting GPLVM-----------------------------------------------------------------------------

# First setting a random seed so we can replicate results
np.random.seed(42)

# Creating a torch tensor for the data set
scaled_df_torch = torch.tensor(scaled_df.sample(n=20000).values, dtype=torch.get_default_dtype())

# Transposing the shape
y = scaled_df_torch.t()

# Initialising the X prior to be a matrix of zeros with the same number of rows as y but with 2 columns 
X_prior_mean = torch.zeros(y.size(1), 2)

# Setting the kernel and initialising the length scale to a vector of ones
kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))

# Cloning the priort so that we don't change it during the course of training
X = Parameter(X_prior_mean.clone())

# Using SparseGPRegression model with num_inducing=32 (default)
# The initial values for Xu are sampled randomly from X_prior_mean
Xu = stats.resample(X_prior_mean.clone(), 32)
gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01), jitter=1e-5)

# Using `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean, 0.1).to_event())
gplvm.autoguide("X", dist.Normal)

# Extracting the losses
losses = gp.util.train(gplvm, num_steps=5000)

# Plotting the losses
plt.plot(losses)
plt.savefig(dim_red_analysis_path + "gplvm_training_losses.png", bbox_inches="tight", dpi=600)

# Extracting the gplvm fitted dimensions
X = pd.DataFrame(gplvm.X_loc.detach().numpy(), columns=["gplvm_dim_0", "gplvm_dim_1"])

# Adding the labels back
X = pd.concat(
    [
        design_name,
        pdb_or_af2,
        dssp_bin,
        isoelectric_point_bin,
        X,
        # organism_scientific_name,
        ],
    axis=1,
)

X.to_csv(dim_red_analysis_path + "gplvm_data.csv", index=False)


# Plotting the gplvm with plotly and overlaying different variables
fig = px.scatter(
    X,
    x="gplvm_dim_0",
    y="gplvm_dim_1",
    color="pdb_or_af2",
    color_discrete_sequence=px.colors.qualitative.G10,
    hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
    opacity=0.6,
)
fig.update_traces(
    marker=dict(size=20, line=dict(width=2)),
    selector=dict(mode="markers"),
)
fig.write_html(dim_red_analysis_path + "gplvm_pdb_or_af2.html")

fig = px.scatter(
    X,
    x="gplvm_dim_0",
    y="gplvm_dim_1",
    color="dssp_bin",
    color_discrete_sequence=px.colors.qualitative.G10,
    hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
    opacity=0.6,
)
fig.update_traces(
    marker=dict(size=20, line=dict(width=2)),
    selector=dict(mode="markers"),
)
fig.write_html(dim_red_analysis_path + "gplvm_dssp_bin.html")

fig = px.scatter(
    X,
    x="gplvm_dim_0",
    y="gplvm_dim_1",
    color="isoelectric_point",
    color_discrete_sequence=px.colors.qualitative.G10,
    hover_data=["design_name", "gplvm_dim_0", "gplvm_dim_1"],
    opacity=0.6,
)
fig.update_traces(
    marker=dict(size=20, line=dict(width=2)),
    selector=dict(mode="markers"),
)
fig.write_html(dim_red_analysis_path + "gplvm_isoelectric_point.html")


