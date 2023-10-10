# 0. Importing packages---------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from scipy.cluster.hierarchy import dendrogram


# 1. Defining helper functions--------------------------------------------


def adj_rand_ind_wssd_plot(data, title, file_name, output_path):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.lineplot(
        data=data,
        x="n_clusters",
        y="weighted_ssd",
        errorbar="sd",
        legend=True,
        ax=ax1,
        color="blue",
        label="Weighted Sum of Square Distances",
    )
    sns.lineplot(
        data=data,
        x="n_clusters",
        y="adj_rand_score",
        errorbar="sd",
        legend=True,
        ax=ax2,
        color="red",
        label="Adjusted Rand Index",
    )
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Weighted Sum of Square Distances")
    ax2.set_ylabel("Adjusted Rand Index")
    x_ticks = range(data["n_clusters"].min(), data["n_clusters"].max() + 1, 2)
    ax1.set_xticks(x_ticks)
    ax2.set_ylim([0, 1.2])
    plt.title(title)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    plt.legend(handles, labels)
    # sns.move_legend(
    #     "upper left",
    #     bbox_to_anchor=(1, 1),
    #     frameon=False,
    # )
    plt.savefig(
        output_path + file_name + ".png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def adj_rand_ind_plot(data, title, file_name, output_path):
    sns.lineplot(
        data=data,
        x="n_clusters",
        y="adj_rand_score",
        # hue=hue,
        errorbar="sd",
        legend=True,
        lw=2,
    )
    plt.xlabel("Number of clusters", fontsize=16)
    plt.ylabel("Adjusted Rand Index", fontsize=16)
    # x_ticks = range(data["n_clusters"].min(), data["n_clusters"].max() + 1, 2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(2, 20)
    plt.ylim([0, 1])
    plt.title(title)
    plt.savefig(
        output_path + file_name + ".png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # hierarchy.set_link_color_palette(sns.color_palette("colorblind", as_cmap=True))

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

    return var_explained_df


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
