# This script implements Principal Component Analysis (PCA)
# across the different data sets and scaling methods.

# 0. Importing packages------------------------------------------------------------
from dim_red_tools import *

# 1. Defining variables----------------------------------------------------------------------------

# Defining the data set list
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
iso_for_contamination_list = [0.0]

# Defining the scaling methods list
# scaling_method_list = ["standard", "robust", "minmax"]
scaling_method_list = ["minmax"]

# Setting random seed
np.random.seed(42)

# Defining number of principal components
n_components = 15


# Defining list of dim ids
dim_ids_list = []
for i in range(0, n_components):
    dim_ids_list.append("dim" + str(i))


# Defining hiver data for plotly
hover_data = ["design_name", "dim0", "dim1"]

# Defining a list of formatted versions of the labels
labels_formatted = [
    # "PDB or AF2",
    "Isoelectric Point",
    "Packing Density",
    "Hydrophobic Fitness",
    "Aggrescan3D Average Value",
    "Designed or Native",
    "Secondary Structure",
    # "Organism",
]

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
            # sns.set_theme(style="darkgrid")
            sns.set_style("ticks")

            # plot_var_list = [
            #     # "pdb_or_af2",
            #     "isoelectric_point",
            #     "packing_density",
            #     "hydrophobic_fitness",
            #     "aggrescan3d_avg_value",
            #     "designed_native",
            #     # "organism_scientific_name",
            # ]

            # # Looping through the different labels that we're interested in
            # for i in range(0, len(plot_var_list)):
            #     var = plot_var_list[i]

            #     if var in [
            #         "isoelectric_point",
            #         "packing_density",
            #         "hydrophobic_fitness",
            #         "aggrescan3d_avg_value",
            #     ]:
            #         cmap = sns.color_palette("viridis", as_cmap=True)

            #     else:
            #         cmap = sns.color_palette("tab10")

            #     plot_pca_plotly(
            #         pca_data=pca_transformed_data.sort_values(by=var, ascending=False),
            #         x="dim0",
            #         y="dim1",
            #         color=var,
            #         hover_data=hover_data,
            #         legend_title=labels_formatted[i],
            #         opacity=0.9,
            #         size=15,
            #         output_path=pca_analysis_path,
            #         file_name="pca_embedding_" + var + ".html",
            #     )

            #     plot_latent_space_2d(
            #         data=pca_transformed_data.sort_values(by=var, ascending=False),
            #         x="dim0",
            #         y="dim1",
            #         axes_prefix="PCA Dim",
            #         legend_title=labels_formatted[i],
            #         hue=var,
            #         # style=var,
            #         alpha=0.8,
            #         s=30,
            #         palette=cmap,
            #         output_path=pca_analysis_path,
            #         file_name="pca_embedding_" + var,
            #     )

            org_list_plant = [
                "Arabidopsis thaliana",
                "Glycine max",
                "Oryza sativa",
                "Zea mays",
            ]

            org_list_bacteria = [
                "Escherichia coli",
                "Salmonella typhimurium",
                "Mycobacterium tuberculosis",
                "Mycobacterium leprae",
                # "Mycobacterium ulcerans",
                # "Neisseria gonorrhoeae",
                # "Staphylococcus aureus",
                # "Streptococcus pneumoniae",
            ]

            org_list_animal = [
                "Homo sapiens",
                "Mus musculus",
                "Rattus norvegicus",
            ]

            org_list_funghi = [
                "Candida albicans",
                "Saccharomyces cerevisiae",
                "Schizosaccharomyces pombe",
                "Madurella mycetomatis",
            ]

            spectral_plot(
                pca_data=pca_transformed_data.sort_values(
                    by="organism_scientific_name", ascending=True
                ),
                group_var="organism_scientific_name",
                value_var_list=dim_ids_list,
                filt_list=org_list_plant,
                title="Plant",
                legend_title="",
                output_path=pca_analysis_path,
                file_name="spectral_plot_plant",
            )

            spectral_plot(
                pca_data=pca_transformed_data.sort_values(
                    by="organism_scientific_name", ascending=True
                ),
                group_var="organism_scientific_name",
                value_var_list=dim_ids_list,
                filt_list=org_list_bacteria,
                title="Bacteria",
                legend_title="",
                output_path=pca_analysis_path,
                file_name="spectral_plot_bacteria",
            )

            spectral_plot(
                pca_data=pca_transformed_data.sort_values(
                    by="organism_scientific_name", ascending=True
                ),
                group_var="organism_scientific_name",
                value_var_list=dim_ids_list,
                filt_list=org_list_animal,
                title="Animal",
                legend_title="",
                output_path=pca_analysis_path,
                file_name="spectral_plot_animal",
            )

            spectral_plot(
                pca_data=pca_transformed_data.sort_values(
                    by="organism_scientific_name", ascending=True
                ),
                group_var="organism_scientific_name",
                value_var_list=dim_ids_list,
                filt_list=org_list_funghi,
                title="Funghi",
                legend_title="",
                output_path=pca_analysis_path,
                file_name="spectral_plot_funghi",
            )

            data_filtered = pca_transformed_data[
                ~pca_transformed_data["dssp_bin"]
                .isin(["Bend", "Hbond Turn", "3 10 Helix"])
                .reset_index(drop=True)
            ]

            # plot_pca_plotly(
            #     pca_data=data_filtered.sort_values(by="dssp_bin", ascending=False),
            #     x="dim0",
            #     y="dim1",
            #     color="dssp_bin",
            #     hover_data=hover_data,
            #     legend_title=labels_formatted[4],
            #     opacity=0.8,
            #     size=15,
            #     output_path=pca_analysis_path,
            #     file_name="pca_embedding_" + "dssp_bin" + ".html",
            # )

            # plot_latent_space_2d(
            #     data=data_filtered.sort_values(by="dssp_bin", ascending=False),
            #     x="dim0",
            #     y="dim1",
            #     axes_prefix="PCA Dim",
            #     legend_title=labels_formatted[5],
            #     hue="dssp_bin",
            #     # style=var,
            #     alpha=0.8,
            #     s=20,
            #     palette=sns.color_palette("tab10"),
            #     output_path=pca_analysis_path,
            #     file_name="pca_embedding_" + "dssp_bin",
            # )
