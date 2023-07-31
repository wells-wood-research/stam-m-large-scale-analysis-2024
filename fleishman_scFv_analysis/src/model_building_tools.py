# This script defines helper functions which are used throughout the
# the model building scripts.

# 0. Importing relevant packages------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve,
    RocCurveDisplay,
    auc,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize, LabelBinarizer
from itertools import chain, cycle


# 1. Defining helper functions---------------------------------------------------------


# Defining a function to evaluate the performance of a classifier
def classifier_eval(
    classifier,
    X_test,
    y_test,
    y_score,
    y_pred,
    cf_mat_list,
    roc_auc_list,
    balanced_accuracy_list,
    precision_list,
    recall_list,
):
    # Generating a confusion matrix on the test set
    cf_mat = confusion_matrix(y_test, classifier.predict(X_test))

    # Calculating weighted roc auc score
    roc_auc = roc_auc_score(
        y_true=y_test, y_score=y_score, average="weighted", multi_class="ovr"
    )

    # Calculating balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)

    # Calculating weighted precision
    precision = precision_score(y_true=y_test, y_pred=y_pred, average="weighted")

    # Calculating weighted recall
    recall = recall_score(y_true=y_test, y_pred=y_pred, average="weighted")

    # Appending to lists for output
    cf_mat_list.append(cf_mat)
    roc_auc_list.append(roc_auc)
    balanced_accuracy_list.append(balanced_accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

    return (
        cf_mat_list,
        roc_auc_list,
        balanced_accuracy_list,
        precision_list,
        recall_list,
    )


# Defining a function to plot confusion matrices
def plot_cf_mat(class_order, cf_mat_total, model_output_path, model_id):
    group_counts = ["{0:0.0f}".format(value) for value in cf_mat_total.flatten()]
    row_sums = cf_mat_total.sum(axis=1)
    group_percentages = [
        "{0:.2%}".format(value)
        for value in (cf_mat_total / row_sums[:, np.newaxis]).flatten()
    ]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)

    ax = sns.heatmap(cf_mat_total, annot=labels, fmt="", cmap="Blues")
    ax.set_xlabel("Predicted Expression Level")
    ax.set_ylabel("Actual Expression Level")

    classes = class_order

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.savefig(
        model_output_path + "cf_mat_" + model_id + ".png", bbox_inches="tight", dpi=600
    )
    plt.savefig(
        model_output_path + "cf_mat_" + model_id + ".svg", bbox_inches="tight", dpi=600
    )
    plt.close()


# Defining a function to plot the roc curves
def plot_roc_curve(
    class_order,
    y_test_binary_list,
    y_score_list,
    model_output_path,
    model_id,
    color_list,
):
    class_mean_tpr_list = []
    class_std_tpr_list = []
    class_mean_fpr_list = []
    class_mean_auc_list = []
    class_std_auc_list = []
    class_stand_err_auc_list = []

    mean_fpr = np.linspace(0, 1, 100)
    all_tprs = []
    all_aucs = []
    class_volume_list = []

    classes = class_order
    for class_id in range(0, len(classes)):
        class_tprs = []
        class_aucs = []
        class_volume = 0

        for i in range(0, len(y_test_binary_list)):
            fpr, tpr, thresholds = roc_curve(
                y_true=y_test_binary_list[i][:, class_id],
                y_score=y_score_list[i][:, class_id],
            )

            # Calculating weighted roc auc score
            roc_auc = roc_auc_score(
                y_true=y_test_binary_list[i][:, class_id],
                y_score=y_score_list[i][:, class_id],
            )

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            class_tprs.append(interp_tpr)
            class_aucs.append(roc_auc)
            all_tprs.append(interp_tpr)
            all_aucs.append(roc_auc)
            class_volume = class_volume + np.sum(y_test_binary_list[i][:, class_id])

        mean_tpr = np.mean(class_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(class_aucs)
        stand_err_auc = 2.576 * (std_auc / np.sqrt(class_volume))
        std_tpr = np.std(class_tprs, axis=0)

        class_mean_tpr_list.append(mean_tpr.tolist())
        class_mean_auc_list.append(mean_auc.tolist())
        class_mean_fpr_list.append(mean_fpr.tolist())
        class_std_auc_list.append(std_auc.tolist())
        class_std_tpr_list.append(std_tpr.tolist())
        class_volume_list.append(class_volume)
        class_stand_err_auc_list.append(stand_err_auc)

    overall_mean_tpr = np.average(
        class_mean_tpr_list, weights=class_volume_list, axis=0
    )
    overall_mean_auc = np.average(class_mean_auc_list, weights=class_volume_list)
    overall_std_auc = np.average(class_std_auc_list, weights=class_volume_list)
    overall_stand_err_auc = np.average(
        class_stand_err_auc_list, weights=class_volume_list
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(0, len(class_mean_fpr_list)):
        sns.lineplot(
            x=class_mean_fpr_list[i],
            y=class_mean_tpr_list[i],
            color=color_list[i],
            label=classes[i]
            + " Expression - Mean ROC (AUC = "
            + "%.2f" % round(class_mean_auc_list[i], 2)
            + " $\pm$ "
            + "%.3f" % round(class_stand_err_auc_list[i], 3)
            + ")",
            lw=3,
            alpha=0.8,
        )

    sns.lineplot(
        x=mean_fpr,
        y=overall_mean_tpr,
        color=color_list[3],
        label="Overall - Mean ROC (AUC = "
        + "%.2f" % round(overall_mean_auc, 2)
        + " $\pm$ "
        + "%.3f" % round(overall_stand_err_auc, 3)
        + ")",
        lw=3,
        alpha=0.8,
    )

    sns.lineplot(
        x=np.linspace(0, 1, 100),
        y=np.linspace(0, 1, 100),
        linestyle="dashed",
        color="black",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [1, 2, 0, 3]

    # add legend to plot
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="lower right",
    )

    # tprs_upper = np.minimum(
    #     np.array(class_mean_tpr_list[i]) + np.array(class_std_tpr_list[i]), 1
    # )
    # tprs_lower = np.maximum(
    #     np.array(class_mean_tpr_list[i]) - np.array(class_std_tpr_list[i]), 0
    # )

    # ax.fill_between(
    #     class_mean_fpr_list[i],
    #     tprs_lower,
    #     tprs_upper,
    #     color=color_list[i],
    #     alpha=0.3,
    #     # label=r"$\pm$ 1 std. dev.",
    # )

    plt.savefig(
        model_output_path + "roc_auc_class_" + model_id + ".png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


# Defining a function to run cross validation on a classifier
def run_cross_validation(
    X,
    y,
    y_binary,
    classifier,
    model_id,
    num_cvs,
    num_folds,
    model_output_path,
):
    # Initialising empty lists to capture information from each fold
    # and a split variable to count each fold
    cf_mat_list = []
    roc_auc_list = []
    y_test_binary_list = []
    y_score_list = []
    balanced_accuracy_list = []
    precision_list = []
    recall_list = []

    for cv in range(0, num_cvs):
        rand_int = np.random.randint(0, high=100000, size=1)[0]
        # Cross validation
        skf = StratifiedKFold(
            n_splits=num_folds,
            random_state=rand_int,
            shuffle=True,
        )

        split_count = 1

        # Looping through each train and test split
        for train_index, test_index in skf.split(X, y):
            # Splitting the train data into a train and test split for cross validation
            X_train = X.filter(items=train_index, axis=0).reset_index(drop=True)
            X_test = X.filter(items=test_index, axis=0).reset_index(drop=True)
            y_train = y.filter(items=train_index, axis=0).reset_index(drop=True)
            y_test = y.filter(items=test_index, axis=0).reset_index(drop=True)
            y_test_binary = y_binary[test_index]

            # Fitting the classifier
            classifier.fit(X_train, y_train)

            # Generating predictions for the test set
            y_score = classifier.predict_proba(X_test)
            y_pred = classifier.predict(X_test)

            y_test_binary_list.append(y_test_binary)
            y_score_list.append(y_score)

            # Evaluating the classifiers performance with different metrics
            (
                cf_mat_list,
                roc_auc_list,
                balanced_accuracy_list,
                precision_list,
                recall_list,
            ) = classifier_eval(
                classifier=classifier,
                X_test=X_test,
                y_test=y_test,
                y_score=y_score,
                y_pred=y_pred,
                cf_mat_list=cf_mat_list,
                roc_auc_list=roc_auc_list,
                balanced_accuracy_list=balanced_accuracy_list,
                precision_list=precision_list,
                recall_list=recall_list,
            )

            split_count = split_count + 1

    for i in range(0, len(cf_mat_list)):
        if i == 0:
            cf_mat_total = cf_mat_list[i]
        else:
            cf_mat_total = cf_mat_total + cf_mat_list[i]

    mean_cv_roc_auc = np.sum(roc_auc_list) / len(roc_auc_list)
    mean_cv_balanced_accuracy = np.sum(balanced_accuracy_list) / len(
        balanced_accuracy_list
    )
    mean_cv_precision = np.sum(precision_list) / len(precision_list)
    mean_cv_recall = np.sum(recall_list) / len(recall_list)

    plot_cf_mat(
        class_order=["Low", "Medium", "High"],
        cf_mat_total=cf_mat_total,
        model_output_path=model_output_path,
        model_id=model_id,
    )

    plot_roc_curve(
        class_order=["High", "Low", "Medium"],
        y_test_binary_list=y_test_binary_list,
        y_score_list=y_score_list,
        model_output_path=model_output_path,
        model_id=model_id,
        color_list=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"],
    )

    # Adding these to a dictionary
    cv_results_dict = {
        "mean_cv_roc_auc": mean_cv_roc_auc,
        "mean_cv_balanced_accuracy": mean_cv_balanced_accuracy,
        "mean_cv_precision": mean_cv_precision,
        "mean_cv_recall": mean_cv_recall,
    }

    # Creating a row data frame
    model_val = pd.DataFrame(
        {
            "model_id": model_id,
            "num_cvs": num_cvs,
            "num_folds": num_folds,
            "mean_cv_roc_auc": cv_results_dict["mean_cv_roc_auc"],
            "mean_cv_balanced_accuracy": cv_results_dict["mean_cv_balanced_accuracy"],
            "mean_cv_precision": cv_results_dict["mean_cv_precision"],
            "mean_cv_recall": cv_results_dict["mean_cv_recall"],
        },
        index=[0],
    )

    return model_val


# Defining a function to fit the naive bayes classifier
def fit_naive_bayes(
    X_train,
    y_train,
    y_train_binary,
    num_cvs,
    num_folds,
    model_output_path,
    scaling_method,
    comp_flag,
    feature_selection,
):
    # Fitting simple model
    classifier = GaussianNB()

    model_id = (
        "naive_bayes_" + scaling_method + "_" + comp_flag + "_" + feature_selection
    )

    model_val = run_cross_validation(
        X=X_train,
        y=y_train,
        y_binary=y_train_binary,
        classifier=classifier,
        model_id=model_id,
        num_cvs=num_cvs,
        num_folds=num_folds,
        model_output_path=model_output_path,
    )

    return model_val
