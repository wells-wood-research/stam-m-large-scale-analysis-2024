# 0. Loading in packages and defining custom functions--------------------------------------------------
import numpy as np
import pandas as pd

# 1. Defining variables---------------------------------------------------------------------------------

# Defining file paths
raw_af2_destress_data_path = "data/raw_data/af2/destress_data_af2.csv"
raw_pdb_destress_data_path = "data/raw_data/pdb/destress_data_pdb.csv"
data_output_path = "data/processed_data/"
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

# 2. Reading in data sets-------------------------------------------------------------------------------

# Reading in raw AF2 DE-STRESS data
raw_af2_destress_data = pd.read_csv(raw_af2_destress_data_path)

# Reading in raw PDB DE-STRESS data
raw_pdb_destress_data = pd.read_csv(raw_pdb_destress_data_path)

# Reading in processed uniprot data
processed_uniprot_data = pd.read_csv(processed_uniprot_data_path)


# 3. Processing data-------------------------------------------------------------------------------------

processed_af2_destress_data = raw_af2_destress_data
processed_pdb_destress_data = raw_pdb_destress_data

processed_af2_destress_data["pdb_or_af2"] = "AF2"
processed_pdb_destress_data["pdb_or_af2"] = "PDB"

# Dropping columns that are not needed. These are dropped because they contain a lot of missing values.
processed_pdb_destress_data.drop(
    [
        "budeff_total",
        "budeff_steric",
        "budeff_steric",
        "budeff_desolvation",
        "budeff_charge",
        "dfire2_total",
    ],
    axis=1,
    inplace=True,
)

# Dropping columns that are not needed. These are dropped because they contain a lot of missing values.
processed_af2_destress_data.drop(
    [
        "budeff_total",
        "budeff_steric",
        "budeff_steric",
        "budeff_desolvation",
        "budeff_charge",
        "dfire2_total",
    ],
    axis=1,
    inplace=True,
)

# Calculating total number of structures that DE-STRESS ran for
num_pdb_structures = processed_pdb_destress_data.shape[0]

# Removing any rows that have NAs in them
processed_pdb_destress_data = processed_pdb_destress_data.dropna(axis=0).reset_index(drop=True)

# Calculating number of structures in the data set after removing missing values
num_pdb_structures_missing_removed = processed_pdb_destress_data.shape[0]

# Calculating how many protein structures are left after removing structures that have missing values for the DE-STRESS metrics.
print(
    "DE-STRESS ran for "
    + str(num_pdb_structures)
    + " PDB structures in total and after removing missing values there are "
    + str(num_pdb_structures_missing_removed)
    + " structures remaining in the data set. This means "
    + str(100 * (round((num_pdb_structures_missing_removed / num_pdb_structures), 4)))
    + "% of the protein structures are covered in this data set."
)

# Joining af2 and pdb destress data sets
processed_destress_data = pd.concat([processed_af2_destress_data, processed_pdb_destress_data]).reset_index(drop=True)

print(processed_destress_data)

# Adding a new field to create a dssp bin
processed_destress_data["dssp_bin"] = np.select(
    [
        processed_destress_data["ss_prop_alpha_helix"].gt(0.5),
        processed_destress_data["ss_prop_beta_bridge"].gt(0.5),
        processed_destress_data["ss_prop_beta_strand"].gt(0.5),
        processed_destress_data["ss_prop_3_10_helix"].gt(0.5),
        processed_destress_data["ss_prop_pi_helix"].gt(0.5),
        processed_destress_data["ss_prop_hbonded_turn"].gt(0.5),
        processed_destress_data["ss_prop_bend"].gt(0.5),
        processed_destress_data["ss_prop_loop"].gt(0.5),
    ],
    ["Mainly alpha helix", "Mainly beta bridge", "Mainly beta strand", "Mainly 3 10 helix", "Mainly pi helix", "Mainly hbond turn", "Mainly bend", "Mainly loop",
     ],
    default="Mixed",
)

# Removing columns 
processed_destress_data.drop(
    [
        "full_sequence",
        "dssp_assignment",
        "ss_prop_alpha_helix",
        "ss_prop_beta_bridge",
        "ss_prop_beta_strand",
        "ss_prop_3_10_helix",
        "ss_prop_pi_helix",
        "ss_prop_hbonded_turn",
        "ss_prop_bend",
        "ss_prop_loop",
        "charge",
    ],
    axis=1,
    inplace=True,
)

# Normalising energy field values by the number of residues
processed_destress_data.loc[:, energy_field_list,] = processed_destress_data.loc[
    :,
    energy_field_list,
].div(processed_destress_data["num_residues"], axis=0)

# Removing columns 
processed_destress_data.drop(
    [
        "num_residues",
        "mass",
        "rosetta_yhh_planarity",
        "rosetta_dslf_fa13",
        "evoef2_interD_total",
        "composition_UNK",
    ],
    axis=1,
    inplace=True,
)


print(processed_destress_data.columns.to_list())

# Defining a column which extracts the uniprot id from the design_name column
processed_destress_data["uniprot_id"] = processed_af2_destress_data["design_name"].str.split("-").str[1]

# Joining on processed uniprot data by uniprot id
processed_destress_data = processed_destress_data.merge(processed_uniprot_data, how="left", left_on="uniprot_id", right_on = "primary_accession")

# Removing columns 
processed_destress_data.drop(
    [
    "primary_accession",
    "uniProtkbId",
    "protein_name",
    "gene_name",
    ],
    axis=1,
    inplace=True,
)

processed_destress_data["organism_scientific_name"] = np.where(processed_destress_data["pdb_or_af2"] == "PDB", "PDB", processed_destress_data["organism_scientific_name"])

# Removing any rows that have NAs in them
processed_destress_data = processed_destress_data.dropna(axis=0).reset_index(drop=True)

processed_destress_data.to_csv(data_output_path + "processed_destress_data.csv", index=False)




