# This script trains a number of different models on the different feature sets
# to predict the Fleishman yeast display expression measure.

# 0. Importing packages---------------------------------------------------------
import numpy as np
import pandas as pd
from model_building_tools import *

# 1. Defining variables---------------------------------------------------------

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining a composition metrics included flag
comp_flag_list = ["comp", "no_comp"]

# Defining a list of feature selection methods
feature_selection_list = ["mi", "rf"]

# Defining feature selection path
feature_selection_path = "fleishman_scFv_analysis/feature_selection/"

# Defining models path
models_path = "fleishman_scFv_analysis/models/"

# Setting the number of cross validation experiments
num_cvs = 10

# Setting the number of cross validation folds
num_folds = 5

# Setting a random seed
np.random.seed(42)

# Creating a data frame to gather the model number, hyper parameters and the validation metrics
model_val_master = pd.DataFrame(
    columns=[
        "model_id",
        "num_cvs",
        "num_folds",
        "mean_cv_roc_auc",
        "mean_cv_balanced_accuracy",
        "mean_cv_precision",
        "mean_cv_recall",
    ]
)

for scaling_method in scaling_method_list:
    for comp_flag in comp_flag_list:
        train_data_path = (
            "fleishman_scFv_analysis/data/processed_data/"
            + scaling_method
            + "/"
            + comp_flag
            + "/"
        )
        feature_selection_scaler_path = (
            feature_selection_path + scaling_method + "/" + comp_flag + "/"
        )
        model_scaler_path = models_path + scaling_method + "/" + comp_flag + "/"

        # 2. Reading in data------------------------------------------------------------

        y_train = pd.read_csv(train_data_path + "y_train.csv")
        y_train_pred_var = y_train["expression_bin_label"]
        y_train_binary = np.load(train_data_path + "y_train_binary.npy")
        for feature_selection in feature_selection_list:
            X_train_scaled = pd.read_csv(train_data_path + "X_train_scaled.csv")
            pdb_scaled = pd.read_csv(train_data_path + "pdb_scaled.csv")
            selected_features = pd.read_csv(
                feature_selection_scaler_path
                + "selected_features_"
                + feature_selection
                + ".csv"
            )
            X_train_scaled_filt = X_train_scaled[
                selected_features["feature_names"]
            ].reset_index(drop=True)
            pdb_scaled_filt = pdb_scaled[
                selected_features["feature_names"]
            ].reset_index(drop=True)

            model_val = fit_naive_bayes(
                X_train=X_train_scaled_filt,
                y_train=y_train_pred_var,
                y_train_binary=y_train_binary,
                num_cvs=num_cvs,
                num_folds=num_folds,
                model_output_path=model_scaler_path,
                scaling_method=scaling_method,
                comp_flag=comp_flag,
                feature_selection=feature_selection,
            )

            model_id = (
                "naive_bayes_"
                + scaling_method
                + "_"
                + comp_flag
                + "_"
                + feature_selection
            )

            # Adding the hyper parameters to the data set
            model_val_master = pd.concat(
                [model_val_master, model_val], axis=0, ignore_index=True
            )
            model_val.to_csv(
                model_scaler_path + "model_val_" + model_id + ".csv",
                index=False,
            )
model_val_master.to_csv(
    models_path + "model_val_master" + ".csv",
    index=False,
)
