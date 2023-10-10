# 0. Importing packages----------------------------------------------------------------------------------
import numpy as np
import pandas as pd


# 1. Defining variables----------------------------------------------------------------------------------

# Defining the file path for the af2 labels
af2_labels_data_path = (
    "pdb_af2_structural_analysis/data/processed_data/af2/iso_for_0.0/minmax/labels.csv"
)

# Output path
output_path = (
    "pdb_af2_structural_analysis/data/processed_data/af2/iso_for_0.0/org_fasta_files/"
)

# 2. Reading in data--------------------------------------------------------------------------------------

# Reading in raw AF2 DE-STRESS data
af2_labels_data = pd.read_csv(af2_labels_data_path)

# Extracting unique oranism list
unique_org_list = af2_labels_data["organism_scientific_name"].unique().tolist()

# Removing NaNs
unique_org_list = [x for x in unique_org_list if str(x) != "nan"]

print(unique_org_list)

# 2. Splitting the data into fasta files by organism

# Looping through the different organisms
for org in unique_org_list:
    # Filtering the data set by the organism
    af2_labels_data_filt = af2_labels_data[
        af2_labels_data["organism_scientific_name"] == org
    ].reset_index(drop=True)

    # Removing spaces from org names
    org = org.replace(" ", "_")

    # Creating a fasta file of all the sequences with their design names
    with open(output_path + str(org) + "_full_seqs.fasta", "w") as f:
        for i in range(0, af2_labels_data_filt.shape[0]):
            f.write(
                ">"
                + af2_labels_data_filt["design_name"].iloc[i]
                + "\n"
                + af2_labels_data_filt["full_sequence"].iloc[i]
                + "\n"
            )
