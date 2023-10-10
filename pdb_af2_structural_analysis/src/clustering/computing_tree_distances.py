# 0. Importing packages and helper functions-------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from scipy.cluster import hierarchy
from clustering_tools import *
import dendropy
import ete3
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram


def get_newick(node, parent_dist, leaf_names, newick="") -> str:
    """
    Convert sciply.cluster.hierarchy.to_tree()-output to Newick format.

    :param node: output of sciply.cluster.hierarchy.to_tree()
    :param parent_dist: output of sciply.cluster.hierarchy.to_tree().dist
    :param leaf_names: list of leaf names
    :param newick: leave empty, this variable is used in recursion.
    :returns: tree in Newick format
    """
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parent_dist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parent_dist - node.dist, newick)
        else:
            newick = ");"
        newick = get_newick(node.get_left(), node.dist, leaf_names, newick=newick)
        newick = get_newick(
            node.get_right(), node.dist, leaf_names, newick=",%s" % (newick)
        )
        newick = "(%s" % (newick)
        return newick


# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination = 0.00

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining the different linkage metrics for hieracrhcial clustering
linkage_list = ["single", "average", "complete", "ward"]

# Defining clustering output path
clustering_overall_results_path = "pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/phylo_tree_comparison_output/"

# 2. Looping through the different data sets------------------------------------------------------

tns = dendropy.TaxonNamespace()

ncbi_phylo_tree = dendropy.Tree.get(
    path="pdb_af2_structural_analysis/data/processed_data/ncbi_phylo_tree.phy",
    schema="newick",
    taxon_namespace=tns,
)
ncbi_phylo_tree.encode_bipartitions()


for dataset in dataset_list:
    for scaling_method in scaling_method_list:
        # Defining the data file path
        processed_data_path = (
            "pdb_af2_structural_analysis/data/processed_data/"
            + dataset
            + "/"
            + "iso_for_"
            + str(iso_for_contamination)
            + "/"
            + scaling_method
            + "/"
        )

        # Defining the pca analysis path
        pca_analysis_path = (
            "pdb_af2_structural_analysis/analysis/clustering/"
            + dataset
            + "/"
            + "iso_for_"
            + str(iso_for_contamination)
            + "/"
            + scaling_method
            + "/"
        )

        # Output path
        clustering_output_path = (
            "pdb_af2_structural_analysis/analysis/clustering/"
            + dataset
            + "/"
            + "iso_for_"
            + str(iso_for_contamination)
            + "/"
            + scaling_method
            + "/"
        )

        # Defining file paths for labels
        labels_df_path = processed_data_path + "labels.csv"

        # 3. Reading in data------------------------------------------------------------------------

        # Reading in processed data
        processed_data = pd.read_csv(
            processed_data_path + "processed_destress_data_scaled.csv"
        )

        # Reading in labels
        labels_df = pd.read_csv(labels_df_path)

        processed_data_joined = pd.concat(
            [
                processed_data,
                labels_df[["organism_scientific_name"]],
            ],
            axis=1,
        )

        # Average each principal component grouped by organism
        processed_data_avg = processed_data_joined.groupby(
            ["organism_scientific_name"], as_index=False
        )[processed_data.columns.to_list()].mean()

        # processed_data_avg.update(
        #     processed_data_avg[["organism_scientific_name"]].applymap("'{}'".format)
        # )

        organism_labels = processed_data_avg["organism_scientific_name"].to_list()

        labels = processed_data_avg[["organism_scientific_name"]]

        processed_data_avg.drop(["organism_scientific_name"], inplace=True, axis=1)

        # 4. Computing trees from avg DE-STRESS data----------------------------------------

        for linkage in linkage_list:
            linkage_matrix = hierarchy.linkage(
                processed_data_avg, method=linkage, metric="euclidean"
            )

            plt.figure(figsize=(9, 8))
            dendrogram(
                linkage_matrix,
                truncate_mode=None,
                labels=organism_labels,
                orientation="left",
                leaf_font_size=8,
            )
            plt.xticks(fontsize=10)
            plt.savefig(
                clustering_overall_results_path
                + "dendrogram_"
                + scaling_method
                + "_"
                + "linkage-"
                + linkage
                + ".png",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close()

            # Convert linkage matrix to dendrogram
            dendro = dendrogram(linkage_matrix, no_plot=True)

            # Convert the linkage matrix to a tree object
            tree = hierarchy.to_tree(linkage_matrix, False)

            # Convert the tree object to the Newick format
            newick = get_newick(tree, tree.dist, organism_labels)

            # Using ETE3
            ete3_tree = ete3.Tree(newick)

            # Save the tree in Newick format
            ete3_tree.write(
                format=1,
                outfile=clustering_overall_results_path
                + "tree_"
                + scaling_method
                + "_"
                + "linkage-"
                + linkage
                + ".nwk",
            )

            # destress_props_tree = dendropy.Tree.get(
            #     path=clustering_overall_results_path
            #     + "tree_"
            #     + scaling_method
            #     + "_"
            #     + "linkage-"
            #     + linkage
            #     + ".nwk",
            #     schema="newick",
            #     taxon_namespace=tns,
            # )

            # destress_props_tree.encode_bipartitions()

            # distance = dendropy.calculate.treecompare.symmetric_difference(
            #     ncbi_phylo_tree, destress_props_tree
            # )
            # print("tree_" + scaling_method + "_" + "linkage-" + linkage)
            # print(f"Robinson-Foulds distance: {distance}")
