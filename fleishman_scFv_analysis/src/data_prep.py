# This script processes the fleishman expression data and structural feature data
# in order to create training and test data sets for machine learning.

# 0. Importing packages------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from data_prep_tools import *

# 1. Defining variables------------------------------------------------------------------------

# Setting the file path for the predictor variable (Fleshman yeast display expression measure)
predictor_data_path = (
    "fleishman_scFv_analysis/data/raw_data/fleishman_antibody_expression_data.csv"
)

# Setting the file path for the features (DE-STRESS metrics)
features_data_path = "fleishman_scFv_analysis/data/raw_data/fleishman_destress_data.csv"

# Setting the data output path
processed_data_path = "fleishman_scFv_analysis/data/processed_data/"

# Setting the data exploration output path
data_exploration_path = "fleishman_scFv_analysis/analysis/data_exploration/"

# Defining a list of DE-STRESS metrics which are energy field metrics
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

# Defining cols to drop
drop_cols = [
    "ss_prop_alpha_helix",
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
    "aggrescan3d_total_value",
    # "rosetta_pro_close",
    # "rosetta_omega",
    # "rosetta_total",
    # "rosetta_fa_rep",
    # "evoef2_total",
    # "evoef2_interS_total",
]

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining a threshold for the spearman correlation coeffient
# in order to remove highly correlated variables
corr_coeff_threshold = 0.7

# Defining a threshold to remove features that have pretty much the same value
constant_features_threshold = 0.8

# Setting variables for train_test_split()
test_size = 0.25
random_state = 42

# 2. Reading in data sets-------------------------------------------------------------------------------

# Reading in Fleishman expression data
raw_predictor_data = pd.read_csv(predictor_data_path)

# Reading in DE-STRESS data
raw_features_data = pd.read_csv(features_data_path)

# 3. Processing predictor data (Fleishman expression data)-----------------------------------------------

predictor_data = raw_predictor_data

# Removing blank spaces in the sequences
predictor_data["sequence"] = predictor_data["sequence"].str.replace(" ", "")

# Adding a new field which extracts the design cycle from the design name
# (excluding 4m5.3 as this is the original template antibody)
predictor_data["design_cycle"] = np.where(
    predictor_data["design_name"] == "4m53",
    "",
    predictor_data["design_name"].str.slice(start=0, stop=1),
)

# Adding a new field which extracts the sequence length
predictor_data["seq_len"] = predictor_data["sequence"].str.len()

# Adding a new field which extracts the target antigen
predictor_data["target_antigen"] = np.where(
    predictor_data["design_name"].str.contains("ins"),
    "Insulin",
    np.where(predictor_data["design_name"].str.contains("acp"), "Tuberculosis ACP", ""),
)

# Adding a new field to create an expression bin
predictor_data["expression_bin"] = pd.cut(predictor_data["expression"], 3)

# Adding a new field to create an sequence length bin
predictor_data["length_bin"] = np.select(
    [
        predictor_data["seq_len"].ge(235) & predictor_data["seq_len"].le(240),
        predictor_data["seq_len"].gt(240) & predictor_data["seq_len"].le(245),
        predictor_data["seq_len"].gt(245) & predictor_data["seq_len"].le(250),
        predictor_data["seq_len"].gt(250) & predictor_data["seq_len"].le(255),
        predictor_data["seq_len"].gt(255) & predictor_data["seq_len"].le(260),
        predictor_data["seq_len"].gt(260) & predictor_data["seq_len"].le(265),
    ],
    ["235-240", "240-245", "245-250", "250-255", "255-260", "260-265"],
    default="Unknown",
)

# Removing 4m53 design
predictor_data = predictor_data[predictor_data["design_name"] != "4m53"].reset_index(
    drop=True
)

# Sorting values by design name
processed_predictor_data = predictor_data.sort_values("design_name").reset_index(
    drop=True
)

# Outputting expression data
processed_predictor_data.to_csv(
    processed_data_path + "processed_expression_data.csv", index=False
)

# 4. Processing features (DE-STRESS metrics)----------------------------------------------------------

features_data = raw_features_data

# Extracting the design name
features_data["design_name"] = features_data["design_name"].str.split("_").str[0]

# Removing 4m53 design
features_data = features_data[features_data["design_name"] != "4m53"]

# Sorting the values by design name
features_data = features_data.sort_values("design_name").reset_index(drop=True)

# Extracting design name so we can join it back on later
design_name_df = features_data["design_name"]

# Normalising energy field values by the number of residues
features_data.loc[
    :,
    energy_field_list,
] = features_data.loc[
    :,
    energy_field_list,
].div(features_data["num_residues"], axis=0)

features_data_columns = features_data.columns.to_list()

# Dropping columns that have been defined manually
features_data_filt = features_data.drop(drop_cols, axis=1)

# Dropping columns that are not numeric
features_data_num = features_data_filt.select_dtypes([np.number]).reset_index(drop=True)

# Printing columns that are dropped because they are not numeric
destress_columns_num = features_data_num.columns.to_list()
dropped_cols_non_num = set(features_data_columns) - set(destress_columns_num)

# Calculating mean and std of features
features_mean_std(
    data=features_data_num,
    output_path=data_exploration_path,
    id="destress_data_num",
)

(
    features_constant_removed,
    constant_features,
) = remove_constant_features(
    data=features_data_num,
    constant_features_threshold=constant_features_threshold,
    output_path=data_exploration_path,
)

print("Features dropped because they're constant")
print(constant_features)


plot_hists_all_columns(
    data=features_constant_removed,
    column_list=features_constant_removed.columns.to_list(),
    output_path=data_exploration_path,
    file_name="/pre_scaling_hist_",
)


# Adding design name back on for the train and test split
features_constant_removed = pd.concat(
    [features_constant_removed, design_name_df], axis=1
)

# 5. Splitting data up into train and test and scaling----------------------------------------------------------

# Scaling features
for scaling_method in scaling_method_list:
    scaled_processed_data_path = processed_data_path + scaling_method + "/"
    scaled_data_exporation_path = data_exploration_path + scaling_method + "/"

    # Train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        features_constant_removed,
        processed_predictor_data,
        test_size=test_size,
        random_state=random_state,
        stratify=processed_predictor_data["expression_bin"],
    )

    # Sorting all the data sets by design name
    X_train = X_train.sort_values("design_name")
    X_test = X_test.sort_values("design_name")
    y_train = y_train.sort_values("design_name")
    y_test = y_test.sort_values("design_name")

    # Exporting data sets as csv files
    X_train.to_csv(scaled_processed_data_path + "X_train.csv", index=False)
    X_test.to_csv(scaled_processed_data_path + "X_test.csv", index=False)
    y_train.to_csv(scaled_processed_data_path + "y_train.csv", index=False)
    y_test.to_csv(scaled_processed_data_path + "y_test.csv", index=False)

    # Saving design name order for training data
    X_train["design_name"].to_csv(
        scaled_processed_data_path + "train_design_name_order.csv", index=False
    )

    # Saving design name order for training data
    X_test["design_name"].to_csv(
        scaled_processed_data_path + "test_design_name_order.csv", index=False
    )

    # Now removing design name from X_train and X_test
    X_train.drop(
        [
            "design_name",
        ],
        axis=1,
        inplace=True,
    )

    # Dropping design name
    X_test.drop(
        [
            "design_name",
        ],
        axis=1,
        inplace=True,
    )

    # Scaling data
    if scaling_method == "minmax":
        # Scaling the data with min max scaler
        scaler = MinMaxScaler().fit(X_train)

    elif scaling_method == "standard":
        # Scaling the data with standard scaler scaler
        scaler = StandardScaler().fit(X_train)

    elif scaling_method == "robust":
        # Scaling the data with robust scaler
        scaler = RobustScaler().fit(X_train)

    # Transforming the data
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
    )
    # Transforming test data with same scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns,
    )

    # Calculating mean and std of features
    features_mean_std(
        data=X_train_scaled,
        output_path=scaled_data_exporation_path,
        id="destress_data_scaled",
    )

    (
        X_train_drop_high_corr,
        drop_cols_high_corr,
    ) = remove_highest_correlators(
        data=X_train_scaled,
        corr_coeff_threshold=corr_coeff_threshold,
        output_path=scaled_data_exporation_path,
    )

    print("Features dropped because of high correlation")
    print(drop_cols_high_corr)

    # Remove these features from test as well
    X_test_drop_high_corr = X_test_scaled.drop(
        drop_cols_high_corr,
        axis=1,
        inplace=False,
    )

    plot_hists_all_columns(
        data=X_train_drop_high_corr,
        column_list=X_train_drop_high_corr.columns.to_list(),
        output_path=scaled_data_exporation_path,
        file_name="/post_scaling_hist_",
    )

    X_train_drop_high_corr.to_csv(
        scaled_processed_data_path + "X_train_scaled.csv",
        index=False,
    )

    X_test_drop_high_corr.to_csv(
        scaled_processed_data_path + "X_test_scaled.csv",
        index=False,
    )
