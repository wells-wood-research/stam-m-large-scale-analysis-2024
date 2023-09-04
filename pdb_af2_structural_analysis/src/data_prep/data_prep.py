# This script is the main script which prepares the PDB and AF2 DE-STRESS data
# so that it is ready for downstream analysis.

# 0. Importing packages and helper functions---------------------------------------------
from data_prep_tools import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 1. Defining variables------------------------------------------------------------------

# Defining the data set list
# dataset_list = ["pdb", "af2", "both"]
dataset_list = ["af2"]

# Defining a list of values for isolation forest
# outlier detection contamination factor
# iso_for_contamination_list = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
iso_for_contamination_list = [0.0]

# Defining the scaling methods list
scaling_method_list = ["standard", "robust", "minmax"]

# Defining file paths for raw data
raw_af2_destress_data_path = (
    "pdb_af2_structural_analysis/data/raw_data/af2/destress_data_af2.csv"
)
raw_pdb_destress_data_path = (
    "pdb_af2_structural_analysis/data/raw_data/pdb/destress_data_pdb.csv"
)

# Defining file path for the de novo protein labels
pdb_denovo_protein_labels_path = (
    "pdb_af2_structural_analysis/data/raw_data/pdb/denovo_pdb_ids.csv"
)

# Defining file paths for the processed uniprot data sets
processed_uniprot_data_af2_path = "pdb_af2_structural_analysis/data/processed_data/uniprot/processed_uniprot_data_af2.csv"
processed_uniprot_data_pdb_path = "pdb_af2_structural_analysis/data/processed_data/uniprot/processed_uniprot_data_pdb.csv"

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
    "uniprot_join_id",
    "aggrescan3d_total_value",
    "rosetta_pro_close",
    "rosetta_omega",
    "rosetta_total",
    "rosetta_fa_rep",
    "evoef2_total",
    "evoef2_interS_total",
]

# Defining the organism groups
organism_animal_list = [
    "Caenorhabditis elegans",
    "Danio rerio",
    "Drosophila melanogaster",
    "Mus musculus",
    "Rattus norvegicus",
    "Homo sapiens",
    "Brugia malayi",
    "Dracunculus medinensis",
    "Onchocerca volvulus",
    "Schistosoma mansoni",
    "Strongyloides stercoralis",
    "Trichuris trichiura",
    "Wuchereria bancrofti",
]

organism_fungi_list = [
    "Candida albicans",
    "Saccharomyces cerevisiae",
    "Cladophialophora carrionii",
    "Fonsecaea pedrosoi",
    "Madurella mycetomatis",
    "Sporothrix schenckii",
    "Ajellomyces capsulatus",
    "Schizosaccharomyces pombe",
    "Paracoccidioides lutzii",
]
organism_bacteria_list = [
    "Escherichia coli",
    "Helicobacter pylori",
    "Campylobacter jejuni",
    "Enterococcus faecium",
    "Klebsiella pneumoniae",
    "Mycobacterium leprae",
    "Mycobacterium tuberculosis",
    "Mycobacterium ulcerans",
    "Neisseria gonorrhoeae",
    "Nocardia brasiliensis",
    "Pseudomonas aeruginosa",
    "Salmonella typhimurium",
    "Shigella dysenteriae",
    "Staphylococcus aureus",
    "Streptococcus pneumoniae",
    "Haemophilus influenzae",
]
organism_plant_list = [
    "Arabidopsis thaliana",
    "Glycine max",
    "Oryza sativa",
    "Zea mays",
]
organism_protozoan_list = [
    "Plasmodium falciparum",
    "Dictyostelium discoideum",
    "Leishmania infantum",
    "Trypanosoma brucei",
    "Trypanosoma cruzi",
]

organism_archaea_list = ["Methanocaldococcus jannaschii"]

organism_other_list = [
    "Other",
    "Unknown",
]

# Defining the labels that we are interested in
labels = [
    "design_name",
    "dssp_bin",
    "pdb_or_af2",
    "charge",
    "isoelectric_point",
    "isoelectric_point_bin",
    "rosetta_total",
    "packing_density",
    "packing_density_bin",
    "hydrophobic_fitness",
    "aggrescan3d_avg_value",
    "aggrescan3d_avg_bin",
    "organism_scientific_name",
    "organism_group",
    "organism_group2",
    "designed_native",
]

# Defining a threshold for the spearman correlation coeffient
# in order to remove highly correlated variables
corr_coeff_threshold = 0.6

# Defining a threshold to remove features that have pretty much the same value
constant_features_threshold = 0.25

# 2. Reading in data sets-------------------------------------------------------------------------------

# Reading in raw AF2 DE-STRESS data
raw_af2_destress_data = pd.read_csv(raw_af2_destress_data_path)

# Reading in raw PDB DE-STRESS data
raw_pdb_destress_data = pd.read_csv(raw_pdb_destress_data_path)

# Reading in processed uniprot data
processed_uniprot_data_af2 = pd.read_csv(processed_uniprot_data_af2_path)
processed_uniprot_data_pdb = pd.read_csv(processed_uniprot_data_pdb_path)

# Renaming data sets
af2_destress_data = raw_af2_destress_data
pdb_destress_data = raw_pdb_destress_data

# Creating a pdb or af2 flag
af2_destress_data["pdb_or_af2"] = "AF2"
pdb_destress_data["pdb_or_af2"] = "PDB"

# Reading in pdb denovo labels
pdb_denovo_protein_labels = pd.read_csv(pdb_denovo_protein_labels_path)

# Lowering the case of the pdb_ids for the de novo design data
pdb_denovo_protein_labels["pdb_id"] = pdb_denovo_protein_labels["pdb_id"].str.lower()

# 3. Joining datasets and removing missing values-----------------------------------------------------------------------------

# Looping through the different data sets
for dataset in dataset_list:
    data_exploration_output_path = (
        "pdb_af2_structural_analysis/analysis/data_exploration/" + dataset + "/"
    )
    data_output_path = (
        "pdb_af2_structural_analysis/data/processed_data/" + dataset + "/"
    )
    # If datasets == "both" then pdb and af2 are concatenated together
    # else we just take the pdb or af2 data
    if dataset == "both":
        destress_data = pd.concat([af2_destress_data, pdb_destress_data]).reset_index(
            drop=True
        )
        drop_cols_all = drop_cols + [
            "primary_accession",
            "pdb_id",
            "organism_scientific_name_pdb",
            "organism_scientific_name_af2",
        ]

    elif dataset == "pdb":
        destress_data = pdb_destress_data
        drop_cols_all = drop_cols + [
            "pdb_id",
            "organism_scientific_name_pdb",
        ]

    elif dataset == "af2":
        destress_data = af2_destress_data
        drop_cols_all = drop_cols + [
            "primary_accession",
            "organism_scientific_name_af2",
        ]

    # Setting the missing value threshold
    if dataset == "pdb":
        missing_val_threshold = 0.2

    else:
        missing_val_threshold = 0.05

    # Removing features that have missing value prop greater than threshold
    destress_data, dropped_cols_miss_vals = remove_missing_val_features(
        data=destress_data,
        output_path=data_exploration_output_path,
        threshold=missing_val_threshold,
    )

    #  Calculating total number of structures that DE-STRESS ran for
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
    num_pdb_structures = (
        destress_data[destress_data["pdb_or_af2"] == "PDB"]
        .reset_index(drop=True)
        .shape[0]
    )
    num_af2_structures = (
        destress_data[destress_data["pdb_or_af2"] == "AF2"]
        .reset_index(drop=True)
        .shape[0]
    )

    print(
        "There are "
        + str(num_pdb_structures)
        + " PDB structures and "
        + str(num_af2_structures)
        + " AF2 structural models."
    )

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
        [
            "Alpha Helix",
            "Beta Bridge",
            "Beta Strand",
            "3 10 Helix",
            "Pi Helix",
            "Hbond Turn",
            "Bend",
            "Loop",
        ],
        default="Mixed",
    )

    # Adding a new field to create a isoelectric point bin
    destress_data["isoelectric_point_bin"] = np.select(
        [
            destress_data["isoelectric_point"].lt(6),
            destress_data["isoelectric_point"].ge(6)
            & destress_data["isoelectric_point"].le(8),
            destress_data["isoelectric_point"].gt(8),
        ],
        [
            "Less than 6",
            "Between 6 and 8",
            "Greater than 8",
        ],
        default="Unknown",
    )

    # Adding a new field to create a packing density bin
    destress_data["packing_density_bin"] = np.select(
        [
            destress_data["packing_density"].lt(40),
            destress_data["packing_density"].ge(40)
            & destress_data["packing_density"].lt(60),
            destress_data["packing_density"].ge(60)
            & destress_data["packing_density"].lt(80),
            destress_data["packing_density"].ge(80),
        ],
        [
            "Less than 40",
            "Between 40 and 60",
            "Between 60 and 80",
            "Greater than 80",
        ],
        default="Unknown",
    )

    # Adding a new field to create a packing density bin
    destress_data["aggrescan3d_avg_bin"] = np.select(
        [
            destress_data["aggrescan3d_avg_value"].lt(-2),
            destress_data["aggrescan3d_avg_value"].ge(-2)
            & destress_data["aggrescan3d_avg_value"].lt(0),
            destress_data["aggrescan3d_avg_value"].ge(0)
            & destress_data["aggrescan3d_avg_value"].lt(2),
            destress_data["aggrescan3d_avg_value"].ge(2),
        ],
        [
            "Less than -2",
            "Between -2 and 0",
            "Between 0 and 2",
            "Greater than 2",
        ],
        default="Unknown",
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

    if dataset == "pdb":
        #  Denovo pdb join id
        destress_data["pdb_denovo_join_id"] = destress_data["design_name"].str.replace(
            "pdb", ""
        )

        # Joining these data sets together
        destress_data_denovo_labels = destress_data.merge(
            pdb_denovo_protein_labels[["pdb_id", "category"]],
            how="left",
            left_on="pdb_denovo_join_id",
            right_on="pdb_id",
        ).reset_index(drop=True)

        # Replacing NaNs in the category column with "native"
        destress_data_denovo_labels["category"] = destress_data_denovo_labels[
            "category"
        ].fillna("Native")

        # Creating a new designed flag that combines two categories
        destress_data_denovo_labels["designed_native"] = np.where(
            destress_data_denovo_labels["category"] == "Native",
            "Native",
            "Designed",
        )

        # Dropping join id
        destress_data_denovo_labels.drop(
            ["pdb_denovo_join_id", "category", "pdb_id"], axis=1, inplace=True
        )

    else:
        destress_data_denovo_labels = destress_data
        destress_data_denovo_labels["designed_native"] = "AF2"

    # Defining a column which extracts the uniprot id from the design_name column
    destress_data_denovo_labels["uniprot_join_id"] = np.where(
        destress_data_denovo_labels["pdb_or_af2"] == "AF2",
        destress_data_denovo_labels["design_name"].str.split("-").str[1],
        destress_data_denovo_labels["design_name"].str[3:8],
    )

    if dataset == "pdb":
        destress_data_uniprot = destress_data_denovo_labels.merge(
            processed_uniprot_data_pdb[["pdb_id", "organism_scientific_name_pdb"]],
            how="left",
            left_on="uniprot_join_id",
            right_on="pdb_id",
        )
        destress_data_uniprot["organism_scientific_name"] = np.where(
            destress_data_uniprot["pdb_or_af2"] == "PDB",
            destress_data_uniprot["organism_scientific_name_pdb"],
            "",
        )
    elif dataset == "af2":
        destress_data_uniprot = destress_data_denovo_labels.merge(
            processed_uniprot_data_af2[
                ["primary_accession", "organism_scientific_name_af2"]
            ],
            how="left",
            left_on="uniprot_join_id",
            right_on="primary_accession",
        )
        destress_data_uniprot["organism_scientific_name"] = np.where(
            destress_data_uniprot["pdb_or_af2"] == "AF2",
            destress_data_uniprot["organism_scientific_name_af2"],
            "",
        )
    else:
        destress_data_uniprot = destress_data_denovo_labels.merge(
            processed_uniprot_data_pdb[["pdb_id", "organism_scientific_name_pdb"]],
            how="left",
            left_on="uniprot_join_id",
            right_on="pdb_id",
        )
        destress_data_uniprot = destress_data_uniprot.merge(
            processed_uniprot_data_af2[
                ["primary_accession", "organism_scientific_name_af2"]
            ],
            how="left",
            left_on="uniprot_join_id",
            right_on="primary_accession",
        )

        destress_data_uniprot["organism_scientific_name"] = np.where(
            destress_data_uniprot["pdb_or_af2"] == "AF2",
            destress_data_uniprot["organism_scientific_name_af2"],
            destress_data_uniprot["organism_scientific_name_pdb"],
        )

    # Dropping duplicates after joining
    destress_data_uniprot.drop_duplicates(inplace=True)

    test = destress_data_uniprot.groupby("organism_scientific_name").count()
    test.to_csv(data_output_path + "unique_organisms.csv")

    # Adding a new field to create an organism group
    destress_data_uniprot["organism_group"] = np.select(
        [
            destress_data_uniprot["organism_scientific_name"].isin(
                organism_animal_list
            ),
            destress_data_uniprot["organism_scientific_name"].isin(organism_fungi_list),
            destress_data_uniprot["organism_scientific_name"].isin(
                organism_bacteria_list
            ),
            destress_data_uniprot["organism_scientific_name"].isin(organism_plant_list),
            destress_data_uniprot["organism_scientific_name"].isin(
                organism_protozoan_list
            ),
            destress_data_uniprot["organism_scientific_name"].isin(
                organism_archaea_list
            ),
            destress_data_uniprot["organism_scientific_name"].isin(organism_other_list),
        ],
        [
            "Animal",
            "Fungi",
            "Bacteria",
            "Plant",
            "Protozoan",
            "Archaea",
            "Other",
        ],
        default="Unknown",
    )

    # Adding a new field to create an organism group
    destress_data_uniprot["organism_group2"] = np.select(
        [
            destress_data_uniprot["organism_scientific_name"].isin(
                organism_animal_list
                + organism_protozoan_list
                + organism_fungi_list
                + organism_plant_list
            ),
            destress_data_uniprot["organism_scientific_name"].isin(
                organism_bacteria_list + organism_archaea_list
            ),
            destress_data_uniprot["organism_scientific_name"].isin(organism_other_list),
        ],
        [
            "Eukaryotes",
            "Prokaryotes",
            "Other",
        ],
        default="Unknown",
    )

    # Normalising energy field values by the number of residues
    destress_data_uniprot.loc[
        :,
        energy_field_list,
    ] = destress_data_uniprot.loc[
        :,
        energy_field_list,
    ].div(destress_data_uniprot["num_residues"], axis=0)

    # Saving labels
    labels_df = save_destress_labels(
        data=destress_data_uniprot,
        labels=labels,
        output_path=data_output_path,
        file_path="labels",
    )

    # 5. Selecting numeric features and filtering----------------------------------------

    destress_columns_full = destress_data_uniprot.columns.to_list()

    # Dropping columns that have been defined manually
    destress_data_uniprot = destress_data_uniprot.drop(drop_cols_all, axis=1)

    # Dropping columns that are not numeric
    destress_data_num = destress_data_uniprot.select_dtypes([np.number]).reset_index(
        drop=True
    )

    # Dropping composition metrics
    destress_data_num = destress_data_num[
        destress_data_num.columns.drop(
            list(destress_data_num.filter(regex="composition"))
        )
    ]

    # Printing columns that are dropped because they are not numeric
    destress_columns_num = destress_data_num.columns.to_list()
    dropped_cols_non_num = set(destress_columns_full) - set(destress_columns_num)

    # Calculating mean and std of features
    features_mean_std(
        data=destress_data_num,
        output_path=data_exploration_output_path,
        id="destress_data_num",
    )

    (
        destress_data_constant_features_removed,
        constant_features,
    ) = remove_constant_features(
        data=destress_data_num,
        constant_features_threshold=constant_features_threshold,
        output_path=data_exploration_output_path,
    )

    print("Features dropped because they're constant")
    print(constant_features)

    # plot_hists_all_columns(
    #     data=destress_data_constant_features_removed,
    #     column_list=destress_data_constant_features_removed.columns.to_list(),
    #     output_path=data_exploration_output_path,
    #     file_name="/pre_scaling_hist_",
    # )

    # 6. Removing outliers-----------------------------------------------------------------

    # Looping through different contamination factors for the isolation forest
    # and scaling mthods
    for iso_for_contamination in iso_for_contamination_list:
        data_exploration_outliers_output_path = (
            data_exploration_output_path + "iso_for_" + str(iso_for_contamination) + "/"
        )
        data_outliers_output_path = (
            data_output_path + "iso_for_" + str(iso_for_contamination) + "/"
        )
        for scaling_method in scaling_method_list:
            data_exploration_scaled_output_path = (
                data_exploration_outliers_output_path + scaling_method + "/"
            )
            data_scaled_output_path = data_outliers_output_path + scaling_method + "/"

            print(data_scaled_output_path)

            if iso_for_contamination == 0.0:
                destress_data_outliers_removed = destress_data_constant_features_removed
                labels_outliers_removed = labels_df

            else:
                (
                    destress_data_outliers_removed,
                    labels_outliers_removed,
                ) = outlier_detection_iso_for(
                    data=destress_data_constant_features_removed,
                    labels=labels_df,
                    contamination=iso_for_contamination,
                    n_estimators=100,
                    max_features=2,
                    output_path=data_exploration_outliers_output_path,
                    file_name="iso_for_outliers",
                )

            destress_data_outliers_removed.to_csv(
                data_outliers_output_path
                + "processed_destress_data_outliers_removed.csv",
                index=False,
            )
            labels_outliers_removed.to_csv(
                data_outliers_output_path + "labels_outliers_removed.csv",
                index=False,
            )

            # plot_hists_all_columns(
            #     data=destress_data_outliers_removed,
            #     column_list=destress_data_outliers_removed.columns.to_list(),
            #     output_path=data_exploration_outliers_output_path,
            #     file_name="/outliers_removed_hist_",
            # )

            # 7. Scaling features--------------------------------------------------------------------

            if scaling_method == "minmax":
                # Scaling the data with min max scaler
                scaler = MinMaxScaler().fit(destress_data_outliers_removed)

            elif scaling_method == "standard":
                # Scaling the data with standard scaler scaler
                scaler = StandardScaler().fit(destress_data_outliers_removed)

            elif scaling_method == "robust":
                # Scaling the data with robust scaler
                scaler = RobustScaler().fit(destress_data_outliers_removed)

            # Transforming the data
            destress_data_scaled = pd.DataFrame(
                scaler.transform(destress_data_outliers_removed),
                columns=destress_data_outliers_removed.columns,
            )

            # Calculating mean and std of features
            features_mean_std(
                data=destress_data_scaled,
                output_path=data_exploration_scaled_output_path,
                id="destress_data_scaled",
            )

            (
                destress_data_remove_high_corr,
                drop_cols_high_corr,
            ) = remove_highest_correlators(
                data=destress_data_scaled,
                corr_coeff_threshold=corr_coeff_threshold,
                output_path=data_exploration_scaled_output_path,
            )

            print("Features dropped because of high correlation")
            print(drop_cols_high_corr)

            # plot_hists_all_columns(
            #     data=destress_data_remove_high_corr,
            #     column_list=destress_data_remove_high_corr.columns.to_list(),
            #     output_path=data_exploration_scaled_output_path,
            #     file_name="/post_scaling_hist_",
            # )

            destress_data_remove_high_corr.to_csv(
                data_scaled_output_path + "processed_destress_data_scaled.csv",
                index=False,
            )

            labels_outliers_removed.to_csv(
                data_scaled_output_path + "labels.csv", index=False
            )
