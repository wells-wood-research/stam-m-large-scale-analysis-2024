# 0. Importing packages---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# 1. Defining variables----------------------------------------------------------------

# Defining file path for PCA latent space
pca_data_path = "data/latent_spaces/pca_data.csv"


# Defining file path for GPLVM latent space
gplvm_data_path = "data/latent_spaces/gplvm_data.csv"


# Defining a file path for the processed DE-STRESS data
processed_destress_data_path = "data/processed_data/processed_destress_data.csv"

# Defining output path for outlier analysis
outlier_analysis_output = "analysis/outlier_analysis/"
pdb_isolation_forest_output = outlier_analysis_output + "isolation_forest/PDB/"


# 2. Reading in data-----------------------------------------------------------------------

processed_destress_data = pd.read_csv(processed_destress_data_path)
print(processed_destress_data.columns.to_list())

# 3. Outlier analysis------------------------------------------------------------------------
processed_destress_data_pdb = processed_destress_data[processed_destress_data["pdb_or_af2"] == "PDB"].reset_index(drop=True)

# All DE-STRESS features
features_pdb = processed_destress_data_pdb.drop(["design_name", "file_name", "pdb_or_af2", "dssp_bin"], axis=1)
iso_for_destress_pdb = IsolationForest(random_state=42, contamination=0.01).fit(features_pdb)
iso_for_destress_pdb_pred = iso_for_destress_pdb.predict(features_pdb)
iso_for_destress_pdb_pred_df = pd.DataFrame(iso_for_destress_pdb_pred, columns=["iso_for_pred"])
processed_destress_data_pdb_iso_for_pred = pd.concat([processed_destress_data_pdb, iso_for_destress_pdb_pred_df], axis=1)

processed_destress_data_pdb_iso_for_pred.to_csv(pdb_isolation_forest_output + "processed_destress_data_pdb_iso_for_pred.csv", index=False)

sns.set_theme(style="darkgrid")

for column in features_pdb.columns.to_list() + ["pdb_or_af2", "dssp_bin"]:

    sns.histplot(data=processed_destress_data_pdb_iso_for_pred, x=column, hue="iso_for_pred", element="step", stat="density", cumulative=True, fill=False, common_norm=False, palette=sns.color_palette("pastel"))

    plt.savefig(pdb_isolation_forest_output + "iso_for_outlier_hist_" + column + ".png")
    plt.close()
