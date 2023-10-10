# 0. Importing packages----------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os


# Function to extract sequence IDs from a FASTA file
def extract_sequence_ids(fasta_file_path):
    ids = []
    with open(fasta_file_path, "r") as fasta_file:
        lines = fasta_file.readlines()
        for line in lines:
            if line.startswith(">"):
                # Remove the ">" character and any leading/trailing whitespace
                seq_id = line[1:].strip()
                ids.append(seq_id)
    return ids


# 1. Defining variables----------------------------------------------------------------------------------

# Defining the file path for the af2 labels
af2_non_redundant_30_path = (
    "pdb_af2_structural_analysis/data/raw_data/af2/af2_non_redundant_30/"
)

# Creating a data frame to gather the results
af2_nonredundant_30_master = pd.DataFrame(
    columns=[
        "design_name",
        "organism_scientific_name",
    ]
)

# Iterate through subfolders
for subdir, _, _ in os.walk(af2_non_redundant_30_path):
    # Check if "clusters_rep_seq.fasta" exists in the current subdirectory
    fasta_file_path = os.path.join(subdir, "clusters_rep_seq.fasta")
    if os.path.isfile(fasta_file_path):
        # Extract sequence IDs from the FASTA file
        sequence_ids = extract_sequence_ids(fasta_file_path)

        # Extract the subdirectory name without the path
        subdirectory_name = os.path.basename(subdir)

        # Extract organsim name
        org_name = subdirectory_name.replace("_full_seqs", "").replace("_", " ")

        print(org_name)

        # Creating a list of the smae org name
        org_name_list = [org_name] * len(sequence_ids)

        # Creating a row data frame
        af2_nonredundant_30 = pd.DataFrame(
            {"design_name": sequence_ids, "organism_scientific_name": org_name_list}
        )

        # Adding the hyper parameters to the data set
        af2_nonredundant_30_master = pd.concat(
            [af2_nonredundant_30_master, af2_nonredundant_30],
            axis=0,
            ignore_index=True,
        )

# # Saving results as a csv file
# af2_nonredundant_30_master.to_csv(
#     af2_non_redundant_30_path + "af2_non_redundant_30_seqs.csv", index=False
# )

print(af2_nonredundant_30_master.groupby("organism_scientific_name").count())
