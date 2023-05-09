# 0. Loading in packages and defining custom functions--------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from data_prep_tools import *
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Defining variables---------------------------------------------------------------------------------

# Defining file paths
raw_af2_destress_data_path = "data/raw_data/af2/destress_data_af2.csv"
raw_pdb_destress_data_path = "data/raw_data/pdb/destress_data_pdb.csv"
data_output_path = "data/processed_data/"
data_exploration_output_path = "analysis/data_exploration/"
processed_uniprot_data_path = "data/processed_data/uniprot_data_df.csv"

# Defining a list of energy field names
energy_field_list = [
    "hydrophobic_fitness",
    "evoef2_total",
    "evoef2_ref_total",
    "evoef2_intraR_total",
    "evoef2_interS_total",
    "evoef2_interD_total",
    "rosetta_total",
    "rosetta_fa_atr",
    "rosetta_fa_rep",
    "rosetta_fa_intra_rep",
    "rosetta_fa_elec",
    "rosetta_fa_sol",
    "rosetta_lk_ball_wtd",
    "rosetta_fa_intra_sol_xover4",
    "rosetta_hbond_lr_bb",
    "rosetta_hbond_sr_bb",
    "rosetta_hbond_bb_sc",
    "rosetta_hbond_sc",
    "rosetta_dslf_fa13",
    "rosetta_rama_prepro",
    "rosetta_p_aa_pp",
    "rosetta_fa_dun",
    "rosetta_omega",
    "rosetta_pro_close",
    "rosetta_yhh_planarity",
]

# Defining the cut off for proportion of missing values 
# (if the feature has more than this then it will be removed)
missing_val_threshold = 0.05
# missing_val_threshold = 0.2

# Defining a low standard deviation threshold
# (if features have less than this threshold then they are removed)
low_std_threshold = 0.02

# Defining a threshold for the spearman correlation coeffient
# in order to remove highly correlated variables
corr_coeff_threshold = 0.75

# Defining dim red labels
dim_red_labels = ["design_name", 
                  "dssp_bin", 
                  "pdb_or_af2", 
                  "charge", 
                  "isoelectric_point", 
                  "rosetta_total", 
                  "packing_density", 
                  "hydrophobic_fitness", 
                  "aggrescan3d_avg_value"]

# Defining cols to drop 
drop_cols = ["ss_prop_alpha_helix", 
             "ss_prop_beta_bridge", 
             "ss_prop_beta_strand",
             "ss_prop_3_10_helix",
             "ss_prop_pi_helix",
             "ss_prop_hbonded_turn",
             "ss_prop_bend",
             "ss_prop_loop",
             "charge",
             "mass",
             "num_residues",
             ]

# drop_cols = ["mass",
#              "num_residues",
#              ]

# Defining scaling method
scaling_method = "robust"

# 2. Reading in data sets-------------------------------------------------------------------------------

# Reading in raw AF2 DE-STRESS data
raw_af2_destress_data = pd.read_csv(raw_af2_destress_data_path)

# Reading in raw PDB DE-STRESS data
raw_pdb_destress_data = pd.read_csv(raw_pdb_destress_data_path)

# Reading in processed uniprot data
processed_uniprot_data = pd.read_csv(processed_uniprot_data_path)

# 3. Joining data sets and removing missing values-------------------------------------------------------------------------------------

af2_destress_data = raw_af2_destress_data
pdb_destress_data = raw_pdb_destress_data

af2_destress_data["pdb_or_af2"] = "AF2"
pdb_destress_data["pdb_or_af2"] = "PDB"

# Joining af2 and pdb destress data sets
destress_data = pd.concat([af2_destress_data, pdb_destress_data]).reset_index(drop=True)
# destress_data = pdb_destress_data

# Removing features that have missing value prop greater than threshold
destress_data, dropped_cols_miss_vals = remove_missing_val_features(data=destress_data, output_path=data_exploration_output_path, threshold=missing_val_threshold)

# Calculating total number of structures that DE-STRESS ran for
num_structures = destress_data.shape[0]

# Now removing any rows that have missing values
destress_data = destress_data.dropna(axis=0).reset_index(drop=True)

# Calculating number of structures in the data set after removing missing values
num_structures_missing_removed = destress_data.shape[0]

# Calculating how many structures are left after removing those with missing values for the DE-STRESS metrics.
print(
    "DE-STRESS ran for "
    + str(num_structures)
    + " PDB and AF2 structures in total and after removing missing values there are "
    + str(num_structures_missing_removed)
    + " structures remaining in the data set. This means "
    + str(100 * (round((num_structures_missing_removed / num_structures), 4)))
    + "% of the protein structures are covered in this data set."
)

# Calculating how many structures for PDB and AF2
num_pdb_structures = destress_data[destress_data["pdb_or_af2"] == "PDB"].reset_index(drop = True).shape[0]
num_af2_structures = destress_data[destress_data["pdb_or_af2"] == "AF2"].reset_index(drop = True).shape[0]

print("There are " + str(num_pdb_structures) + " PDB structures and " + str(num_af2_structures) + " AF2 structural models.")


print(destress_data.columns.to_list())

print("Features dropped because of missing values")
print(dropped_cols_miss_vals)

# 4. Creating new fields and saving labels--------------------------------------------------------------------------------

# Adding a new field to create a dssp bin
destress_data["dssp_bin"] = np.select(
    [
        destress_data["ss_prop_alpha_helix"].gt(0.5),
        destress_data["ss_prop_beta_bridge"].gt(0.5),
        destress_data["ss_prop_beta_strand"].gt(0.5),
        destress_data["ss_prop_3_10_helix"].gt(0.5),
        destress_data["ss_prop_pi_helix"].gt(0.5),
        destress_data["ss_prop_hbonded_turn"].gt(0.5),
        destress_data["ss_prop_bend"].gt(0.5),
        destress_data["ss_prop_loop"].gt(0.5),
    ],
    ["Mainly alpha helix", "Mainly beta bridge", "Mainly beta strand", "Mainly 3 10 helix", "Mainly pi helix", "Mainly hbond turn", "Mainly bend", "Mainly loop",
     ],
    default="Mixed",
)

# # Adding a new field to create a dssp bin
# destress_data["isoelectric_point_bin"] = np.select(
#     [
#         destress_data["isoelectric_point"].lt(6),
#         destress_data["isoelectric_point"].ge(6) and destress_data["isoelectric_point"].le(8),
#         destress_data[destress_data]
#         destress_data["isoelectric_point"].gt(6),
#     ],
#     ["1-5", "6-8", "9-13"],
#     default="Other",
# )



# Normalising energy field values by the number of residues
destress_data.loc[:, energy_field_list,] = destress_data.loc[
    :,
    energy_field_list,
].div(destress_data["num_residues"], axis=0)

# Saving labels
save_destress_labels(data=destress_data, labels=dim_red_labels, output_path=data_output_path, file_path="labels_pdb")

# 5. Scaling features--------------------------------------------------------------

destress_columns_full = destress_data.columns.to_list()

# Dropping columns that have been defined manually 
destress_data = destress_data.drop(drop_cols, axis=1)

# Dropping columns that are not numeric
destress_data_num = destress_data.select_dtypes([np.number]).reset_index(drop=True)

# Printing columns that are dropped because they are not numeric
destress_columns_num = destress_data_num.columns.to_list()
dropped_cols_non_num = set(destress_columns_full) - set(destress_columns_num)

# Calculating mean and std of features
features_mean_std(data=destress_data_num, output_path=data_exploration_output_path, id="destress_data_num")

if scaling_method == "minmax":

    # Scaling the data with min max scaler
    scaler = MinMaxScaler().fit(destress_data_num)

elif scaling_method == "standard":

    # Scaling the data with standard scaler scaler
    scaler = StandardScaler().fit(destress_data_num)

elif scaling_method == "robust":

    # Scaling the data with robust scaler
    scaler = RobustScaler().fit(destress_data_num)

# Transforming the data
destress_data_scaled = pd.DataFrame(
    scaler.transform(destress_data_num), columns=destress_data_num.columns
)

# for col in destress_data_num.columns.to_list():

#     sns.histplot(data=destress_data_num, x=col)
#     plt.savefig(data_exploration_output_path + "/before_scaling/hist_" + col + ".png")
#     plt.close()

# for col in destress_data_scaled.columns.to_list():

#     sns.histplot(data=destress_data_scaled, x=col)
#     plt.savefig(data_exploration_output_path + "/after_minmax_scaling/hist_" + col + ".png")
#     plt.close()

# Calculating mean and std of features
features_mean_std(data=destress_data_scaled, output_path=data_exploration_output_path, id="destress_data_scaled")


# 5. Filtering features-----------------------------------------------------------------

destress_data_filt, drop_cols_low_std, drop_cols_high_corr= filter_features(data=destress_data_scaled, 
                                                                            low_std_threshold=low_std_threshold,
                                                                            corr_coeff_threshold=corr_coeff_threshold)

print(destress_data_filt)
print(destress_data_filt.columns.to_list())

print("Features dropped because of low std")
print(drop_cols_low_std)
print("Features dropped because of high correlation")
print(drop_cols_high_corr)

# # Defining a column which extracts the uniprot id from the design_name column
# destress_data["uniprot_id"] = processed_af2_destress_data["design_name"].str.split("-").str[1]

# # Joining on processed uniprot data by uniprot id
# destress_data = destress_data.merge(processed_uniprot_data, how="left", left_on="uniprot_id", right_on = "primary_accession")

# # Removing columns 
# destress_data.drop(
#     [
#     "primary_accession",
#     "uniProtkbId",
#     # "protein_name",
#     "gene_name",
#     ],
#     axis=1,
#     inplace=True,
# )

# destress_data["uniprot_id"] = np.where(destress_data["pdb_or_af2"] == "PDB", "PDB", destress_data["uniprot_id"])
# destress_data["organism_scientific_name"] = np.where(destress_data["pdb_or_af2"] == "PDB", "PDB", destress_data["organism_scientific_name"])
# destress_data["protein_name"] = np.where(destress_data["pdb_or_af2"] == "PDB", "PDB", destress_data["protein_name"])

# # Removing any rows that have NAs in them
# destress_data = destress_data.dropna(axis=0).reset_index(drop=True)

destress_data_filt.to_csv(data_output_path + "processed_destress_data.csv", index=False)






