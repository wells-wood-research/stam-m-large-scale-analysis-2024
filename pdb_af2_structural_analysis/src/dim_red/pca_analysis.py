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
scaling_method_list = ["standard", "robust", "minmax"]
# scaling_method_list = ["minmax"]

# Setting random seed
np.random.seed(42)

# Defining number of principal components
n_components = 8

# Defining list of dim ids
dim_ids_list = []
for i in range(0, n_components):
    dim_ids_list.append("dim" + str(i))


# Defining hiver data for plotly
hover_data = ["design_name", "dim0", "dim1"]

# Defining a list of formatted versions of the labels
labels_formatted = [
    # "PDB or AF2",
    # "Isoelectric Point",
    # "Packing Density",
    # "Hydrophobic Fitness",
    # "Aggrescan3D Average Value",
    # "Designed or Native",
    # "Secondary Structure",
    "Organism",
]

# 2. Looping through the different data sets------------------------------------------------------

for dataset in dataset_list:
    for iso_for_contamination in iso_for_contamination_list:
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
                "pdb_af2_structural_analysis/analysis/dim_red/pca/"
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

            plot_var_list = [
                # "pdb_or_af2",
                # "isoelectric_point",
                # "packing_density",
                # "hydrophobic_fitness",
                # "aggrescan3d_avg_value",
                # "designed_native",
                "organism_scientific_name",
            ]

            # Looping through the different labels that we're interested in
            for i in range(0, len(plot_var_list)):
                var = plot_var_list[i]

                if var in [
                    "isoelectric_point",
                    "packing_density",
                    "hydrophobic_fitness",
                    "aggrescan3d_avg_value",
                ]:
                    cmap = sns.color_palette("viridis", as_cmap=True)

                else:
                    cmap = sns.color_palette("colorblind")

                # plot_pca_plotly(
                #     pca_data=pca_transformed_data.sort_values(by=var, ascending=False),
                #     x="dim0",
                #     y="dim1",
                #     color=var,
                #     hover_data=hover_data,
                #     legend_title=labels_formatted[i],
                #     opacity=0.9,
                #     size=15,
                #     output_path=pca_analysis_path,
                #     file_name="pca_embedding_" + var + ".html",
                # )

                # plot_latent_space_2d(
                #     data=pca_transformed_data.sort_values(by=var, ascending=False),
                #     x="dim0",
                #     y="dim1",
                #     axes_prefix="PCA Dim",
                #     legend_title=labels_formatted[i],
                #     hue=var,
                #     hue_order=pca_transformed_data.sort_values(by=var, ascending=False)[
                #         var
                #     ]
                #     .unique()
                #     .tolist(),
                #     # style=var,
                #     alpha=0.8,
                #     s=70,
                #     palette=cmap,
                #     output_path=pca_analysis_path,
                #     file_name="pca_embedding_" + var,
                # )

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

            # spectral_plot(
            #     pca_data=pca_transformed_data.sort_values(
            #         by="organism_scientific_name", ascending=True
            #     ),
            #     group_var="organism_scientific_name",
            #     value_var_list=dim_ids_list,
            #     filt_list=org_list_plant,
            #     title="Plant",
            #     legend_title="",
            #     output_path=pca_analysis_path,
            #     file_name="spectral_plot_plant",
            #     palette=sns.color_palette("colorblind"),
            # )

            # spectral_plot(
            #     pca_data=pca_transformed_data.sort_values(
            #         by="organism_scientific_name", ascending=True
            #     ),
            #     group_var="organism_scientific_name",
            #     value_var_list=dim_ids_list,
            #     filt_list=org_list_bacteria,
            #     title="Bacteria",
            #     legend_title="",
            #     output_path=pca_analysis_path,
            #     file_name="spectral_plot_bacteria",
            #     palette=sns.color_palette("colorblind"),
            # )

            # spectral_plot(
            #     pca_data=pca_transformed_data.sort_values(
            #         by="organism_scientific_name", ascending=True
            #     ),
            #     group_var="organism_scientific_name",
            #     value_var_list=dim_ids_list,
            #     filt_list=org_list_animal,
            #     title="Animal",
            #     legend_title="",
            #     output_path=pca_analysis_path,
            #     file_name="spectral_plot_animal",
            #     palette=sns.color_palette("colorblind"),
            # )

            # spectral_plot(
            #     pca_data=pca_transformed_data.sort_values(
            #         by="organism_scientific_name", ascending=True
            #     ),
            #     group_var="organism_scientific_name",
            #     value_var_list=dim_ids_list,
            #     filt_list=org_list_funghi,
            #     title="Funghi",
            #     legend_title="",
            #     output_path=pca_analysis_path,
            #     file_name="spectral_plot_funghi",
            #     palette=sns.color_palette("colorblind"),
            # )

            # data_filtered = pca_transformed_data[
            #     ~pca_transformed_data["dssp_bin"]
            #     .isin(["Bend", "Hbond Turn", "3 10 Helix"])
            #     .reset_index(drop=True)
            # ]

            # plot_pca_plotly(
            #     pca_data=data_filtered.sort_values(by="dssp_bin", ascending=False),
            #     x="dim0",
            #     y="dim1",
            #     color="dssp_bin",
            #     hover_data=hover_data,
            #     legend_title=labels_formatted[0],
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
            #     legend_title="Secondary Structure",
            #     hue="dssp_bin",
            #     hue_order=["Alpha Helix", "Beta Strand", "Loop", "Mixed"],
            #     # style=var,
            #     alpha=0.8,
            #     s=70,
            #     palette=sns.color_palette("colorblind"),
            #     output_path=pca_analysis_path,
            #     file_name="pca_embedding_" + "dssp_bin",
            # )

            # Plotting histograms of PC1 and PC2

            for var in [
                "designed_native",
                "dssp_bin",
                "organism_group",
                "isoelectric_point_bin",
                "packing_density_bin",
                "aggrescan3d_avg_bin",
            ]:
                if var == "designed_native":
                    pca_transformed_data_filt = pca_transformed_data

                    hue_order = ["Designed", "Native"]
                    legend_title = "Designed or Native"

                elif var == "dssp_bin":
                    pca_transformed_data_filt = pca_transformed_data[
                        ~pca_transformed_data["dssp_bin"]
                        .isin(["Bend", "Hbond Turn", "3 10 Helix"])
                        .reset_index(drop=True)
                    ]
                    hue_order = ["Alpha Helix", "Beta Strand", "Loop", "Mixed"]
                    legend_title = "Secondary Structure"

                elif var == "organism_group":
                    pca_transformed_data_filt = pca_transformed_data

                    hue_order = ["Animal", "Bacteria", "Funghi", "Plant", "Other"]
                    legend_title = "Organism"

                elif var == "isoelectric_point_bin":
                    pca_transformed_data_filt = pca_transformed_data

                    hue_order = ["Less than 6", "Between 6 and 8", "Greater than 8"]
                    legend_title = "Isoelectric Point"

                elif var == "packing_density_bin":
                    pca_transformed_data_filt = pca_transformed_data

                    hue_order = [
                        "Less than 40",
                        "Between 40 and 60",
                        "Between 60 and 80",
                        "Greater than 80",
                    ]
                    legend_title = "Packing Density"

                elif var == "aggrescan3d_avg_bin":
                    pca_transformed_data_filt = pca_transformed_data

                    hue_order = [
                        "Less than -2",
                        "Between -2 and 0",
                        "Between 0 and 2",
                        "Greater than 2",
                    ]
                    legend_title = "Aggrescan3D Average Score"

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 8))

                sns.histplot(
                    data=pca_transformed_data_filt,
                    x="dim0",
                    hue=var,
                    hue_order=hue_order,
                    element="poly",
                    stat="density",
                    common_norm=False,
                    cumulative=True,
                    fill=False,
                    lw=5,
                    legend=False,
                    ax=ax1,
                    palette=sns.color_palette("colorblind"),
                    # kde=True,
                )
                sns.histplot(
                    data=pca_transformed_data_filt,
                    x="dim1",
                    hue=var,
                    hue_order=hue_order,
                    element="poly",
                    stat="density",
                    common_norm=False,
                    cumulative=True,
                    fill=False,
                    lw=5,
                    legend=True,
                    ax=ax2,
                    palette=sns.color_palette("colorblind"),
                    # kde=True,
                )
                sns.histplot(
                    data=pca_transformed_data_filt,
                    x="dim2",
                    hue=var,
                    hue_order=hue_order,
                    element="poly",
                    stat="density",
                    common_norm=False,
                    cumulative=True,
                    fill=False,
                    lw=5,
                    legend=False,
                    ax=ax3,
                    palette=sns.color_palette("colorblind"),
                    # kde=True,
                )
                sns.histplot(
                    data=pca_transformed_data_filt,
                    x="dim3",
                    hue=var,
                    hue_order=hue_order,
                    element="poly",
                    stat="density",
                    common_norm=False,
                    cumulative=True,
                    fill=False,
                    lw=5,
                    legend=False,
                    ax=ax4,
                    palette=sns.color_palette("colorblind"),
                    # kde=True,
                )
                ax1.set_xlabel("PC1")
                ax2.set_xlabel("PC2")
                ax3.set_xlabel("PC3")
                ax4.set_xlabel("PC4")
                sns.move_legend(
                    ax2,
                    "upper left",
                    bbox_to_anchor=(1, 1),
                    frameon=False,
                    title=legend_title,
                )
                plt.savefig(
                    pca_analysis_path + "pca_hist_" + var + ".png",
                    bbox_inches="tight",
                    dpi=600,
                )
                plt.close()
