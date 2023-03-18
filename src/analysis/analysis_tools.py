# 0. Importing packages---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import decomposition

# 1. Defining helper functions for PCA------------------

# Defining a function to calculate the variance explained by a 
# number of different principal components
def pca_var_explained(data, n_components_list, output_path):

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
        output_path + "pca_var_explained.csv",
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
    plt.savefig(output_path + "pca_var_explained.png")
    plt.close()


def perform_pca(data, labels_df, n_components, output_path):

    # Performing PCA
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data)

    # Saving contributions of the features to the principal components
    pca_feat_contr_to_cmpts = pd.DataFrame(
        np.round(abs(pca.components_), 4), columns=data.columns
    )
    pca_feat_contr_to_cmpts.to_csv(output_path + "feat_contr_to_cmpts.csv", index=True)

    # Selecting the 10 largest contributers to pca component 1
    pca_components_1_contr = pca_feat_contr_to_cmpts.iloc[0].nlargest(10, keep="first")
    pca_components_1_contr.to_csv(output_path + "components_1_contr.csv", index=True)

    # Selecting the 10 largest contributers to pca component 2
    pca_components_2_contr = pca_feat_contr_to_cmpts.iloc[1].nlargest(10, keep="first")
    pca_components_2_contr.to_csv(output_path + "components_2_contr.csv", index=True)

    # Selecting the 10 largest contributers to pca component 2
    pca_components_3_contr = pca_feat_contr_to_cmpts.iloc[2].nlargest(10, keep="first")
    pca_components_3_contr.to_csv(output_path + "components_3_contr.csv", index=True)

    # Transforming the data
    pca_transformed_data = pca.transform(data)

    # Converting to data frame and renaming columns
    pca_transformed_data = pd.DataFrame(pca_transformed_data).rename(
        columns={0: "dim0", 1: "dim1", 2: "dim2", 3: "dim3", 4: "dim4", 5: "dim5", 6: "dim6", 7: "dim7", 8: "dim8", 9: "dim9"}
    )

    # Adding the labels back
    pca_transformed_data = pd.concat(
        [labels_df, pca_transformed_data], axis=1
    )

    # Outputting the transformed data
    pca_transformed_data.to_csv(
        output_path
        + "pca_transformed_data.csv",
        index=False,
    )

    return pca_transformed_data


def plot_pca_2d(pca_data, x, y, hue, alpha, s, output_path, file_name): 

    sns.color_palette("tab10")

    xlabel = str(int(x[-1]) + 1)
    ylabel = str(int(y[-1]) + 1)

    # PCA 2d scatter plot
    plot= sns.scatterplot(
        x=x,
        y=y,
        data=pca_data,
        hue=hue,
        # style=style,
        alpha=alpha,
        s=s,
    )
    # plt.title("PCA of designs by model type and structure id")
    plt.xlabel("Principal Component " + xlabel)
    plt.ylabel("Principal Component " + ylabel)
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    plot.get_legend().set_title(None)
    plt.savefig(
        output_path + file_name + xlabel + ylabel + ".png",
        bbox_inches="tight",
        dpi=600,
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

