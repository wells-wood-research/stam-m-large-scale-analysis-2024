# This script implements Principal Component Analysis (PCA)
# across the different data sets and scaling methods.

# 0. Importing packages------------------------------------------------------------
from dim_red_tools import *

# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination_list = [0.03, 0.04, 0.05]

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Setting random seed
np.random.seed(42)

# Defining number of principal components
n_components = 20

# Defining hiver data for plotly
hover_data = ["design_name", "dim0", "dim1", "dim2"]

# 2. Looping through the different data sets------------------------------------------------------

for dataset in dataset_list:
    for iso_for_contamination in iso_for_contamination_list:
        for scaling_method in scaling_method_list:
            # Defining the data file path
            processed_data_path = (
                "data/processed_data/"
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
                "analysis/dim_red/pca/"
                + dataset
                + "/"
                + "iso_for_"
                + str(iso_for_contamination)
                + "/"
                + scaling_method
                + "/"
            )

            # Defining the path for processed AF2 DE-STRESS data
            processed_destress_data_path = (
                processed_data_path + "processed_destress_data_scaled.csv"
            )

            # Defining file paths for labels
            labels_df_path = processed_data_path + "labels.csv"

            # 3. Reading in data------------------------------------------------------------------------

            # Defining the path for the processed AF2 DE-STRESS data
            processed_destress_data = pd.read_csv(processed_destress_data_path)

            # Reading in labels
            labels_df = pd.read_csv(labels_df_path)

            # 4. Performing PCA--------------------------------------------------------------------------

            pca_var_explained(
                data=processed_destress_data,
                n_components=n_components,
                file_name="pca_var_explained",
                output_path=pca_analysis_path,
            )

            pca_transformed_data = perform_pca(
                data=processed_destress_data,
                labels_df=labels_df,
                n_components=n_components,
                output_path=pca_analysis_path,
                file_path="pca_transformed_data",
                components_file_path="comp_contrib",
            )

            # 5. Plotting 2d spaces---------------------------------------------------------------------

            # Setting theme for plots
            sns.set_theme(style="darkgrid")

            # Looping through the different labels that we're interested in
            for i in range(0, len(labels_df.columns.to_list())):
                var = labels_df.columns.to_list()[i]

                if var != "design_name":
                    if var in [
                        "isoelectric_point",
                        "charge",
                        "rosetta_total",
                        "packing_density",
                        "hydrophobic_fitness",
                        "aggrescan3d_avg_value",
                        "organism_scientific_name",
                    ]:
                        cmap = sns.color_palette("viridis", as_cmap=True)

                    else:
                        cmap = sns.color_palette("tab10")

                    #         plot_pca_plotly(
                    #             pca_data=pca_transformed_data,
                    #             x="dim0",
                    #             y="dim1",
                    #             color=var,
                    #             hover_data=hover_data,
                    #             opacity=0.8,
                    #             size=5,
                    #             output_path=pca_analysis_path,
                    #             file_name="pca_embedding_" + var + ".html",
                    #         )

                    if var != "organism_scientific_name":
                        plot_latent_space_2d(
                            data=pca_transformed_data,
                            x="dim0",
                            y="dim1",
                            axes_prefix="PCA Dim",
                            legend_title=var,
                            hue=var,
                            # style=var,
                            alpha=0.8,
                            s=20,
                            palette=cmap,
                            output_path=pca_analysis_path,
                            file_name="pca_embedding_" + var,
                        )
