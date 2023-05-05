# 0. Loading in packages and defining custom functions--------------------------------------------------
import numpy as np
import pandas as pd
from scipy import stats

# 1. Filtering out features-----------------------------------------------------------------------------

# Defining a function which removes features with missing values
# if the proportion is greater than a threshold
def remove_missing_val_features(data, output_path, threshold):

    data_missing_count = data.isnull().sum()
    data_missing_count_df = pd.DataFrame(data_missing_count, columns=["num_missing"])
    data_missing_count_df["prop_missing"] = data_missing_count_df["num_missing"]/data.shape[0]
    data_missing_count_df.to_csv(output_path + "destress_data_missing_count_df.csv")

    features_to_remove = data_missing_count_df[data_missing_count_df["prop_missing"] > threshold].index.values

    new_data = data.drop(features_to_remove, axis=1)

    return new_data, features_to_remove

# Defining a function which saves csv files of the labels 
def save_destress_labels(data, labels, output_path, file_path):

    data_filt = data[labels]
    data_filt.to_csv(output_path + file_path + ".csv", index=False)

# Defining a function to compute mean and std of features
def features_mean_std(data, output_path, id):

    data_std = data.std().sort_values(ascending=False)
    data_mean = data.mean().sort_values(ascending=False)

    data_std.to_csv(output_path + "data_std_" + id + ".csv")
    data_mean.to_csv(output_path + "data_mean_" + id + ".csv")

def remove_highest_correlators(data, corr_coeff_threshold):

    drop_cols_high_corr = []

    data_corr = stats.spearmanr(data)
    data_corr_df = pd.DataFrame(data_corr[0], columns=data.columns.to_list(), index=data.columns.to_list())
    data_corr_df_abs = data_corr_df.abs()
    data_cutoff_count = data_corr_df_abs[data_corr_df_abs > corr_coeff_threshold].count().sort_values(ascending=False)

    while data_cutoff_count.max() > 1:

        data_corr = stats.spearmanr(data)
        data_corr_df = pd.DataFrame(data_corr[0], columns=data.columns.to_list(), index=data.columns.to_list())
        data_corr_df_abs = data_corr_df.abs()
        data_cutoff_count = data_corr_df_abs[data_corr_df_abs > corr_coeff_threshold].count().sort_values(ascending=False)
        data_cutoff_count_max = data_cutoff_count.index.values[0]
        drop_cols_high_corr.append(data_cutoff_count_max)

        data = data.drop(data_cutoff_count_max, axis=1)

    data_new = data

    return data_new, drop_cols_high_corr


# Defining a function to gilter out further features
def filter_features(data, low_std_threshold, corr_coeff_threshold):

    # First removing columns that have low standard deviation
    data_std = data.std().sort_values(ascending=False)
    drop_cols_low_std = data_std[data_std < low_std_threshold].index.values
    
    # Filtering these columns out
    data = data.drop(drop_cols_low_std, axis=1)
    
    # Now removing features that are highly correlated using spearman 
    # correlation coefficient
    data_new, drop_cols_high_corr = remove_highest_correlators(data, corr_coeff_threshold)

    return data_new, drop_cols_low_std, drop_cols_high_corr
