# This script provides helper functions for the dimensionality reduction
# analysis.

# 0. Importing packages--------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import decomposition
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances

# 1. Defining helper functions for Principal Component Analysis (PCA)---------------------------------


# Defining a function to calculate the variance explained by a
# number of different principal components
def pca_var_explained(data, n_components, file_name, output_path):
    # Performing PCA with the specified components
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data)

    # Calculating the variance explained
    var_explained = pca.explained_variance_ratio_

    # Calculating the cumulative sum
    var_explained_sum = np.cumsum(var_explained)

    # Calculating list of components
    components_list = range(0, n_components, 1)

    # Creating dict
    var_explained_dict = {
        "n_components": components_list,
        "var_explained": var_explained,
        "var_explained_sum": var_explained_sum,
    }

    # Appending to the data frame
    var_explained_df = pd.DataFrame(var_explained_dict)

    # Saving as a csv file
    var_explained_df.to_csv(
        output_path + file_name + ".csv",
        index=False,
    )

    sns.set_theme(style="darkgrid")

    # Plotting the data and saving
    sns.lineplot(
        x="n_components",
        y="var_explained_sum",
        data=var_explained_df,
    )
    plt.title("""Cumulative variance explained by number of principal components""")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance explained")
    plt.savefig(output_path + file_name + ".png")
    plt.close()


# Defining a script to perform Principal Component Analysis (PCA)
# for a data set and a specified number of principal components
def perform_pca(
    data, labels_df, n_components, output_path, file_path, components_file_path
):
    # Performing PCA
    pca_model = decomposition.PCA(n_components=n_components)
    pca_model.fit(data)

    # Saving contributions of the features to the principal components
    pca_feat_contr_to_cmpts = pd.DataFrame(
        np.round(abs(pca_model.components_), 4), columns=data.columns
    )

    pca_feat_contr_to_cmpts.to_csv(
        output_path + components_file_path + "_feat_contr_to_cmpts.csv", index=True
    )

    # Defining the columns dict
    columns_dict = {}

    # Selecting the 10 largest contributers to each principal component
    for i in range(0, n_components):
        pca_components_contr = pca_feat_contr_to_cmpts.iloc[i].nlargest(
            10, keep="first"
        )
        pca_components_contr.to_csv(
            output_path + components_file_path + str(i) + "_contr.csv", index=True
        )

        columns_dict[i] = "dim" + str(i)

    # Transforming the data
    pca_transformed_data = pca_model.transform(data)

    # Converting to data frame and renaming columns
    pca_transformed_data = pd.DataFrame(pca_transformed_data).rename(
        columns=columns_dict
    )

    if labels_df is None:
        pca_transformed_data = pca_transformed_data

    else:
        # Adding the labels back
        pca_transformed_data = pd.concat([labels_df, pca_transformed_data], axis=1)

    # Outputting the transformed data
    pca_transformed_data.to_csv(
        output_path + file_path + ".csv",
        index=False,
    )

    return pca_transformed_data


def plot_latent_space_2d(
    data,
    x,
    y,
    axes_prefix,
    legend_title,
    hue,
    alpha,
    s,
    palette,
    output_path,
    file_name,
):
    sns.color_palette("tab10")

    x_id = str(int(x[-1]) + 1)
    y_id = str(int(y[-1]) + 1)

    # PCA 2d scatter plot
    plot = sns.scatterplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        # style=style,
        alpha=alpha,
        palette=palette,
        s=s,
        legend=True,
    )
    plt.xlabel(axes_prefix + " " + x_id)
    plt.ylabel(axes_prefix + " " + y_id)
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    # plot.get_legend().set_title(legend_title)
    plt.savefig(
        output_path + file_name + x_id + y_id + ".png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_pca_plotly(
    pca_data, x, y, color, hover_data, opacity, size, output_path, file_name
):
    fig = px.scatter(
        pca_data,
        x=x,
        y=y,
        # z="pca_dim2",
        color=color,
        color_discrete_sequence=px.colors.qualitative.G10,
        hover_data=hover_data,
        opacity=opacity,
    )
    fig.update_traces(
        marker=dict(size=size, line=dict(width=2)),
        selector=dict(mode="markers"),
    )
    fig.write_html(output_path + file_name)


def distance_to_reference(data, dim_columns, feature, distance_metric, output_path):
    # Filtering
    pca_data_filt = data[dim_columns]

    # Computing distances
    distances = pairwise_distances(X=pca_data_filt, metric=distance_metric, n_jobs=1)

    # Converting to a data frame
    distances_df = pd.DataFrame(distances, columns=data[feature].to_list())

    distances_df = distances_df.round(decimals=4)

    # Outputting as a csv
    distances_df.to_csv(
        output_path + "pca" + str(len(dim_columns)) + "d_distances_all.csv", index=False
    )

    return distances_df
