# 0. Importing packages----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


def extract_boundary_points(data, slope, x, y, intercept, ineq):

    latent_space_boundary = data


    y_calc = slope * data[x] + intercept

    if ineq == "g":

        latent_space_boundary = latent_space_boundary[latent_space_boundary[y] > y_calc]

    elif ineq == "l":

        latent_space_boundary = latent_space_boundary[latent_space_boundary[y] < y_calc]


    latent_space_boundary = latent_space_boundary.reset_index(drop=True)


    return latent_space_boundary


def compute_boundary_line(x1, y1, x2, y2):

    delta_y = y2 - y1
    delta_x = x2 - x1

    slope = delta_y/delta_x

    intercept = (y1 - slope * x1)

    return slope, intercept

# 1. Defining variables-----------------------------------------------

# Defining file path for PCA latent space
pca_data_path = "analysis/dim_red/pca/pca_transformed_data.csv"


# Defining file path for GPLVM latent space
gplvm_data_path = "analysis/dim_red/gplvm/gplvm_data.csv"


# Defining a file path for the processed DE-STRESS data
processed_destress_data_path = "data/processed_data/processed_destress_data.csv"

# Defining a file path for the latent space output
latent_space_output_path = "data/latent_spaces/"

# 2. Reading in data--------------------------------------------------

pca_data = pd.read_csv(pca_data_path).rename(columns={"dim0": "pca_dim0", "dim1": "pca_dim1"})
gplvm_data = pd.read_csv(gplvm_data_path).rename(columns={"dim0": "gplvm_dim0", "dim1": "gplvm_dim1"})
processed_destress_data = pd.read_csv(processed_destress_data_path)

# # 3. Exploring PCA latent space----------------------------------------

# # Filtering for PDB structures
# pca_data_pdb = pca_data[pca_data["pdb_or_af2"] == "PDB"].reset_index(drop=True)

# # Computing boundary lines
# slope1, intercept1 = compute_boundary_line(x1=-0.57, y1=-0.45, x2=-0.21, y2=0.59)
# slope2, intercept2 = compute_boundary_line(x1=-0.59, y1=-0.32, x2=0.50, y2=-0.32)
# slope3, intercept3 = compute_boundary_line(x1=0.45, y1=-0.32, x2=0.07, y2=0.56)
# slope4, intercept4 = compute_boundary_line(x1=-0.29, y1=0.41, x2=0.25, y2=0.41)

# pca_pdb_latent_space_boundary1 = extract_boundary_points(data=pca_data_pdb, slope=slope1, x="pca_dim0", y="pca_dim1", intercept=intercept1, ineq="g")
# pca_pdb_latent_space_boundary2 = extract_boundary_points(data=pca_data_pdb, slope=slope2, x="pca_dim0", y="pca_dim1", intercept=intercept2, ineq="l")
# pca_pdb_latent_space_boundary3 = extract_boundary_points(data=pca_data_pdb, slope=slope3, x="pca_dim0", y="pca_dim1", intercept=intercept3, ineq="g")
# pca_pdb_latent_space_boundary4 = extract_boundary_points(data=pca_data_pdb, slope=slope4, x="pca_dim0", y="pca_dim1", intercept=intercept4, ineq="g")

# x_filt1 = pca_data_pdb["pca_dim0"][pca_data_pdb["pca_dim0"] < -0.1][pca_data_pdb["pca_dim0"] > -0.65]
# x_filt2 = pca_data_pdb["pca_dim0"][pca_data_pdb["pca_dim0"] > -0.2]


# pca_pdb_latent_space_boundary1_metrics = processed_destress_data[processed_destress_data.design_name.isin(pca_pdb_latent_space_boundary1.design_name)]
# pca_pdb_latent_space_boundary2_metrics = processed_destress_data[processed_destress_data.design_name.isin(pca_pdb_latent_space_boundary2.design_name)]
# pca_pdb_latent_space_boundary3_metrics = processed_destress_data[processed_destress_data.design_name.isin(pca_pdb_latent_space_boundary3.design_name)]
# pca_pdb_latent_space_boundary4_metrics = processed_destress_data[processed_destress_data.design_name.isin(pca_pdb_latent_space_boundary4.design_name)]

# pca_pdb_latent_space_boundary1_metrics["boundary"] = "boundary1"
# pca_pdb_latent_space_boundary2_metrics["boundary"] = "boundary2"
# pca_pdb_latent_space_boundary3_metrics["boundary"] = "boundary3"
# pca_pdb_latent_space_boundary4_metrics["boundary"] = "boundary4"

# pca_pdb_latent_space_boundary_metrics = pd.concat([pca_pdb_latent_space_boundary1_metrics, pca_pdb_latent_space_boundary2_metrics, pca_pdb_latent_space_boundary3_metrics, pca_pdb_latent_space_boundary4_metrics])
# pca_pdb_latent_space_boundary_metrics.reset_index(drop=True, inplace=True)

# # PCA 2d scatter plot
# sns.scatterplot(
#     x="pca_dim0",
#     y="pca_dim1",
#     data=pca_data_pdb,
#     # hue="Yeast Display Expression",
#     # hue_order=["Low", "Medium", "High", "PDB"],
#     # style="Design Cycle",
#     alpha=0.75,
#     s=150,
#     legend=False,
# )
# sns.lineplot(
#     x=x_filt1,
#     y=slope1*x_filt1 + intercept1,
#     color="black",

# )
# sns.lineplot(
#     x=pca_data_pdb["pca_dim0"],
#     y=slope2*pca_data_pdb["pca_dim0"] + intercept2,
#     color="black",

# )
# sns.lineplot(
#     x=x_filt2,
#     y=slope3*x_filt2 + intercept3,
#     color="black",

# )
# sns.lineplot(
#     x=pca_data_pdb["pca_dim0"],
#     y=slope4*pca_data_pdb["pca_dim0"] + intercept4,
#     color="black",

# )
# plt.savefig(latent_space_output_path+"boundary_lines_pdb.png")
# plt.close()

# # Computing histograms for each feature and boundary areas
# features = processed_destress_data.drop(["design_name", "file_name", "pdb_or_af2", "dssp_bin"], axis=1)

# for column in features.columns.to_list():

#     sns.histplot(data=pca_pdb_latent_space_boundary_metrics, x=column, hue="boundary", element="poly", stat="density")

#     plt.savefig(latent_space_output_path + "pdb_boundary_hists_" + column + ".png")
#     plt.close()




# Computing mutual information--------------------------------------------------

# features = processed_destress_data.drop(["design_name", "file_name", "pdb_or_af2", "dssp_bin"], axis=1)

# mutual_info_pca_dim0 = mutual_info_regression(X=features, y=pca_data["pca_dim0"])
# mutual_info_pca_dim1 = mutual_info_regression(X=features, y=pca_data["pca_dim1"])

# mutual_info_gplvm_dim0 = mutual_info_regression(X=features, y=gplvm_data["gplvm_dim_0"])
# mutual_info_gplvm_dim1 = mutual_info_regression(X=features, y=gplvm_data["gplvm_dim_1"])

# mutual_info_pca_dim0.to_csv("data/latent_spaces/mutual_info_pca_dim0.csv")
# mutual_info_pca_dim1.to_csv("data/latent_spaces/mutual_info_pca_dim1.csv")
# mutual_info_gplvm_dim0.to_csv("data/latent_spaces/mutual_info_gplvm_dim0.csv")
# mutual_info_gplvm_dim1.to_csv("data/latent_spaces/mutual_info_gplvm_dim1.csv")


# Computing correlation coefficient

# features = processed_destress_data.drop(["design_name", "file_name", "pdb_or_af2", "dssp_bin"], axis=1)

joined_df = pd.concat([processed_destress_data, pca_data[["pca_dim0", "pca_dim1"]], gplvm_data[["gplvm_dim0", "gplvm_dim1"]]], axis=1)

print(joined_df)

# corr = joined_df.corr().abs()

# corr_pca_dim0 = corr["pca_dim0"].sort_values(ascending=False)
# corr_pca_dim1 = corr["pca_dim1"].sort_values(ascending=False)
# corr_gplvm_dim0 = corr["gplvm_dim0"].sort_values(ascending=False)
# corr_gplvm_dim1 = corr["gplvm_dim1"].sort_values(ascending=False)

# corr_pca_dim0.to_csv("data/latent_spaces/corr_coef_pca_dim0.csv")
# corr_pca_dim1.to_csv("data/latent_spaces/corr_coef_pca_dim1.csv")
# corr_gplvm_dim0.to_csv("data/latent_spaces/corr_coef_gplvm_dim0.csv")
# corr_gplvm_dim1.to_csv("data/latent_spaces/corr_coef_gplvm_dim1.csv")

corr_spearman = stats.spearmanr(joined_df)
corr_spearman_df = pd.DataFrame(corr_spearman[0], columns=joined_df.columns.to_list(), index=joined_df.columns.to_list())
corr_spearman_df_abs = corr_spearman_df.abs()

corr_pca_dim0 = corr_spearman_df_abs["pca_dim0"].sort_values(ascending=False)
corr_pca_dim1 = corr_spearman_df_abs["pca_dim1"].sort_values(ascending=False)
corr_gplvm_dim0 = corr_spearman_df_abs["gplvm_dim0"].sort_values(ascending=False)
corr_gplvm_dim1 = corr_spearman_df_abs["gplvm_dim1"].sort_values(ascending=False)

corr_pca_dim0.to_csv("data/latent_spaces/corr_coef_pca_dim0.csv")
corr_pca_dim1.to_csv("data/latent_spaces/corr_coef_pca_dim1.csv")
corr_gplvm_dim0.to_csv("data/latent_spaces/corr_coef_gplvm_dim0.csv")
corr_gplvm_dim1.to_csv("data/latent_spaces/corr_coef_gplvm_dim1.csv")

# print(corr_coef_pca_dim0)
# print(corr_coef_pca_dim1)
# print(corr_coef_gplvm_dim0)
# print(corr_coef_gplvm_dim1)




