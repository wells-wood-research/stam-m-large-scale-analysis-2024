# This script provides helper functions for the organism phylo tree
# analysis

# 0. Importing packages--------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram

# 1. Defining helper functions for Principal Component Analysis (PCA)---------------------------------


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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_organism_groups(data, output_path, title, legend_title):
    sns.color_palette("tab10")

    # PCA 2d scatter plot
    plot = sns.scatterplot(
        x="dim0",
        y="dim1",
        data=data,
        hue="organism_group",
        # style=style,
        alpha=0.7,
        s=50,
        legend=True,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    plot.get_legend().set_title(legend_title)
    plt.savefig(
        output_path + "avg_pc1_vs_pc2_organism.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_organism_groups_plotly(data, output_path, title):
    sns.color_palette("tab10")

    fig = px.scatter_3d(
        data,
        x="dim0",
        y="dim1",
        z="dim2",
        color="organism_group",
        color_discrete_sequence=px.colors.qualitative.G10,
        hover_data=[
            "organism_scientific_name",
            "organism_group",
            "dim0",
            "dim1",
            "dim2",
        ],
        title=title,
        opacity=0.8,
    )
    fig.update_traces(
        marker=dict(size=10, line=dict(width=2)),
        selector=dict(mode="markers"),
    )
    fig.write_html(output_path + "avg_pc1_vs_pc2_vs_pc3_organism_plotly.html")
