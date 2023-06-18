# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import decomposition
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

# 1. Defining helper functions for PCA------------------

# Defining a function to calculate the variance explained by a 
# number of different principal components
def pca_var_explained(data, n_components_list, file_name, output_path):

    # Creating a data frame to collect results
    var_explained_df = pd.DataFrame(columns=["n_components", "var_explained"])

    # Looping through different scaling methods and components
    for n_components in n_components_list:

        # Performing PCA with the specified components
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(data)

        # Calculating the variance explained
        var_explained = np.sum(pca.explained_variance_ratio_)

        # Appending to the data frame
        var_explained_df = var_explained_df.append(
            {"n_components": n_components, "var_explained": var_explained},
            ignore_index=True,
        )

    # Saving as a csv file
    var_explained_df.to_csv(
        output_path + file_name + ".csv",
        index=False,
    )

    sns.set_theme(style="darkgrid")

    # Plotting the data and saving
    sns.lineplot(
        x="n_components",
        y="var_explained",
        data=var_explained_df,
    )
    plt.title("""Variance explained by number of pca components.""")
    plt.xlabel("Number of components")
    plt.savefig(output_path + file_name + ".png")
    plt.close()


def perform_pca(data, labels_df, n_components, output_path, file_path, components_file_path):

    # Performing PCA
    pca_model = decomposition.PCA(n_components=n_components)
    pca_model.fit(data)

    # Saving contributions of the features to the principal components
    pca_feat_contr_to_cmpts = pd.DataFrame(
        np.round(abs(pca_model.components_), 4), columns=data.columns
    )
    pca_feat_contr_to_cmpts.to_csv(output_path + components_file_path + "_feat_contr_to_cmpts.csv", index=True)

    # Selecting the 10 largest contributers to pca component 1
    pca_components_1_contr = pca_feat_contr_to_cmpts.iloc[0].nlargest(10, keep="first")
    pca_components_1_contr.to_csv(output_path + components_file_path + "_1_contr.csv", index=True)

    # Selecting the 10 largest contributers to pca component 2
    pca_components_2_contr = pca_feat_contr_to_cmpts.iloc[1].nlargest(10, keep="first")
    pca_components_2_contr.to_csv(output_path + components_file_path + "_2_contr.csv", index=True)

    # # Selecting the 10 largest contributers to pca component 2
    # pca_components_3_contr = pca_feat_contr_to_cmpts.iloc[2].nlargest(10, keep="first")
    # pca_components_3_contr.to_csv(output_path + components_file_path + "_3_contr.csv", index=True)

    # Transforming the data
    pca_transformed_data = pca_model.transform(data)

    # Converting to data frame and renaming columns
    pca_transformed_data = pd.DataFrame(pca_transformed_data).rename(
        columns={0: "dim0", 1: "dim1", 2: "dim2", 3: "dim3", 4: "dim4", 5: "dim5", 6: "dim6", 7: "dim7", 8: "dim8", 9: "dim9", 10: "dim10", 11: "dim11", 12: "dim12", 13: "dim13", 14: "dim14", 15: "dim15", 16: "dim16", 17: "dim17", 18: "dim18", 19: "dim19"}
    )

    if labels_df is None:

        pca_transformed_data = pca_transformed_data

    else:

        # Adding the labels back
        pca_transformed_data = pd.concat(
            [labels_df, pca_transformed_data], axis=1
        )

    # Outputting the transformed data
    pca_transformed_data.to_csv(
        output_path
        + file_path
        + ".csv",
        index=False,
    )

    return pca_transformed_data, pca_model


def plot_latent_space_2d(data, x, y, axes_prefix, legend_title, hue, alpha, s, palette, output_path, file_name): 

    sns.color_palette("tab10")

    x_id = str(int(x[-1]) + 1)
    y_id = str(int(y[-1]) + 1)

    # PCA 2d scatter plot
    plot= sns.scatterplot(
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


def plot_pca_plotly(pca_data, x, y, color, hover_data, opacity, size, output_path, file_name): 

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


def spectral_plot(data, x, y, ax, metric, output_path):

    sns.set_theme(style="darkgrid")
    sns.color_palette("tab10")
    
    # Plot the pca spectre for the different sequences
    sns.lineplot(x=x, 
                 y=y,
                 hue=metric,
                 errorbar="sd",
                 legend="full",
                 ax=ax,
                 data=data)
    # ax.legend(
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.3),
    #     ncol=1,
    # )
    ax.set_xlabel("Principal Component ID")
    ax.set_ylabel("Principal Component Value")
#     plt.savefig(
#     output_path + file_name + ".png",
#     bbox_inches="tight",
#     dpi=600,
# )
    # plt.close()


def distance_to_reference(data, dim_columns, distance_metric, output_path):

    # Filtering
    pca_data_filt = data[dim_columns]

    # Computing distances
    distances = pairwise_distances(X=pca_data_filt, metric=distance_metric, n_jobs=1)

    # Converting to a data frame
    distances_df = pd.DataFrame(distances, columns = data["organism_scientific_name"].to_list())

    distances_df = distances_df.round(decimals=4)

    # Outputting as a csv
    distances_df.to_csv(output_path + "pca" + str(len(dim_columns)) + "d_distances_all.csv", index=False)

    return distances_df


# Defining a function to perform outlier detection
def outlier_detection_iso_for(data, labels, contamination, n_estimators, max_features, output_path, file_name):

    # Isolation Forest
    iso_for = IsolationForest(random_state=42, n_estimators=n_estimators, max_features=max_features, contamination=contamination).fit(data)
    iso_for_pred = iso_for.predict(data)
    iso_for_pred_df = pd.DataFrame(iso_for_pred, columns=["iso_for_pred"])

    # Outputting outliers to a csv file
    iso_for_outliers = pd.concat([data, labels, iso_for_pred_df], axis=1)
    iso_for_outliers = iso_for_outliers[iso_for_outliers["iso_for_pred"] == -1].reset_index(drop=True)
    iso_for_outliers.to_csv(output_path + file_name + str(contamination) + ".csv", index=False)

    # data_outliers_removed = features[iso_for_pred_df["iso_for_pred"] != -1].reset_index(drop=True)
    # labels_outliers_removed = labels[iso_for_pred_df["iso_for_pred"] != -1].reset_index(drop=True)

    return iso_for_pred_df


def phylo_tree_from_dist_matrix(distance_matrix, organism_names):

    # Create a DistanceMatrix object from the distance matrix and organism names
    dist_matrix = DistanceMatrix(organism_names, matrix=distance_matrix)

    # Use the DistanceTreeConstructor to build the tree
    constructor = DistanceTreeConstructor()
    tree = constructor.upgma(dist_matrix)

    # Print the resulting tree
    Phylo.draw(tree)

    return tree


