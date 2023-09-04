# 0. Importing packages----------------------------------------------------------------------------------
import numpy as np
import pandas as pd


# 1. Defining variables----------------------------------------------------------------------------------

# Defining the file path for the raw af2 data
raw_af2_destress_data_path = (
    "pdb_af2_structural_analysis/data/raw_data/af2/destress_data_af2.csv"
)

# 2. Reading in data--------------------------------------------------------------------------------------

# Reading in raw AF2 DE-STRESS data
raw_af2_destress_data = pd.read_csv(raw_af2_destress_data_path)


print(raw_af2_destress_data["full_sequence"])


uniq_seqs = raw_af2_destress_data["full_sequence"].nunique()
total_seqs = raw_af2_destress_data["full_sequence"].count()

print(uniq_seqs)
print(total_seqs)
