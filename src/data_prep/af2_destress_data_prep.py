# 0. Loading in packages and defining custom functions--------------------------------------------------
import numpy as np
import pandas as pd

# 1. Defining variables---------------------------------------------------------------------------------

# Defining file paths
raw_af2_destress_data_path = "data/raw_data/af2/destress_data_af2.csv"
data_output_path = "data/processed_data/"

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

raw_af2_destress_data = pd.read_csv(raw_af2_destress_data_path)
# print(raw_af2_destress_data)
# print(raw_af2_destress_data.columns.to_list())

# 3. Processing data-------------------------------------------------------------------------------------

processed_af2_destress_data = raw_af2_destress_data

# Normalising energy field values by the number of residues
processed_af2_destress_data.loc[:, energy_field_list,] = processed_af2_destress_data.loc[
    :,
    energy_field_list,
].div(processed_af2_destress_data["num_residues"], axis=0)

# Defining a pdb or af2 flag and labelling all of these proteins as af2
processed_af2_destress_data["pdb_or_af2_flag"] = "AF2"

# Defining a column which extracts the uniprot id from the design_name column
processed_af2_destress_data["uniprot_id"] = processed_af2_destress_data["design_name"].str.split("-").str[1]

print(processed_af2_destress_data["uniprot_id"])

processed_af2_destress_data.to_csv(data_output_path + "processed_af2_destress_data.csv", index=False)




