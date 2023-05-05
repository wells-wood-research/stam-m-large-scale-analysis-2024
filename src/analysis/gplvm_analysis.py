# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import os

import torch
from torch.nn import Parameter
import pyro 
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

from analysis_tools import *

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(1)

# 1. Defining variables----------------------------------

print(os.getcwd())

# Defining the data file path
processed_data_path = "data/processed_data/"

# Defining the path for processed AF2 DE-STRESS data
processed_destress_data_path = processed_data_path + "processed_destress_data.csv"

# Defining file paths for labels
labels_df_path = processed_data_path + "labels.csv"

# Defining the output paths
gplvm_analysis_path = "analysis/dim_red/gplvm/"

# Setting random seed
np.random.seed(42)

# Defining hyperparameters for gplvm
gplvm_input_dim = 2
gplvm_lengthscale = torch.ones(2)
gplvm_kernel = gp.kernels.RBF(input_dim=gplvm_input_dim, lengthscale=gplvm_lengthscale)
gplvm_noise = torch.tensor(0.01)
gplvm_inducing_points=32
gplvm_jitter = 1e-5
gplvm_train_num_steps = 5000

# 1. Reading in data--------------------------------------

# Defining the path for the processed AF2 DE-STRESS data
processed_destress_data = pd.read_csv(processed_destress_data_path)

# Reading in labels
labels_df = pd.read_csv(labels_df_path)

# Reading in GPLVM data
gplvm_transformed_data = pd.read_csv("analysis/dim_red/gplvm/gplvm_data.csv")

# 2. Fitting GPLVM-----------------------------------------------------------------------------

# # First setting a random seed so we can replicate results
# np.random.seed(42)

# # Creating a torch tensor for the data set
# scaled_df_torch = torch.tensor(processed_destress_data.sample(n=1000).values, dtype=torch.get_default_dtype())

# # Transposing the shape
# y = scaled_df_torch.t()

# gplvm_X_prior_mean = torch.zeros(y.size(1), 2)

# # Cloning the priort so that we don't change it during the course of training
# X = Parameter(gplvm_X_prior_mean.clone())

# # Using SparseGPRegression model with num_inducing=32 (default)
# # The initial values for Xu are sampled randomly from X_prior_mean
# Xu = stats.resample(gplvm_X_prior_mean.clone(), gplvm_inducing_points)
# gplvm = gp.models.SparseGPRegression(X=X, y=y, kernel=gplvm_kernel, Xu=Xu, noise=gplvm_noise, jitter=gplvm_jitter)

# # Using `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
# gplvm.X = pyro.nn.PyroSample(dist.Normal(gplvm_X_prior_mean, 0.1).to_event())
# gplvm.autoguide("X", dist.Normal)

# # Extracting the losses
# losses = gp.util.train(gplvm, num_steps=gplvm_train_num_steps)

# # Plotting the losses
# plt.plot(losses)
# plt.savefig(gplvm_analysis_path + "gplvm_training_losses.png", bbox_inches="tight", dpi=600)
# plt.close()

# # Extracting the gplvm fitted dimensions
# X = pd.DataFrame(gplvm.X_loc.detach().numpy(), columns=["dim0", "dim1"])

# # Adding the labels back
# gplvm_transformed_data = pd.concat(
#     [ labels_df,
#       X,
#     ],
#     axis=1,
# )

# gplvm_transformed_data.to_csv(gplvm_analysis_path + "gplvm_transformed_data.csv", index=False)

sns.set_theme(style="darkgrid")

labels_formatted = ['Design Name', 'Secondary Structure', 'PDB or AF2', 'Charge', 'Isoelectric Point', 'Rosetta Total Score', 'Packing Density', 'Hydrophobic Fitness', 'Aggrescan3d Average Score']

# Plotting
for i in range(0, len(labels_df.columns.to_list())):

    var = labels_df.columns.to_list()[i]
    label = labels_formatted[i]

    if var != "design_name":

        if var in ["isoelectric_point", "charge", "rosetta_total", "packing_density", "hydrophobic_fitness", "aggrescan3d_avg_value"]:

            cmap = sns.color_palette("viridis", as_cmap=True)

        else:

            cmap= sns.color_palette("tab10")

        # plot_pca_plotly(pca_data=pca_transformed_data, 
        #                 x="dim0", 
        #                 y="dim1", 
        #                 color=var, 
        #                 hover_data=pca_hover_data, 
        #                 opacity=0.6, 
        #                 size=20, 
        #                 output_path=pca_analysis_path, 
        #                 file_name="pca_embedding_" + var + ".html")
        
        plot_latent_space_2d(data=gplvm_transformed_data[gplvm_transformed_data["pdb_or_af2"] == "AF2"].reset_index(drop=True), 
                    x="dim0",
                    y="dim1",
                    axes_prefix = "GPLVM Dim",
                    legend_title=label,
                    hue=var,
                    # style=var,
                    alpha=0.9, 
                    s=10,
                    palette=cmap,
                    output_path=gplvm_analysis_path, 
                    file_name="gplvm_embedding_af2_" + var)
        
        plot_latent_space_2d(data=gplvm_transformed_data, 
                    x="dim0", 
                    y="dim1",
                    axes_prefix = "GPLVM Dim",
                    legend_title=label,
                    hue=var,
                    # style=var,
                    alpha=0.9, 
                    s=10,
                    palette=cmap,
                    output_path=gplvm_analysis_path, 
                    file_name="gplvm_embedding_" + var)