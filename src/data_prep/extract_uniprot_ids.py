# 0. Loading in packages and defining custom functions--------------------------------------------------
import numpy as np
import pandas as pd

# 1. Defining variables---------------------------------------------------------------------------------

# Defining file paths
raw_af2_destress_data_path = "data/raw_data/af2/destress_data_af2.csv"
raw_uniprot_data_path = "data/raw_data/uniprot/"

# 2. Reading in data sets-------------------------------------------------------------------------------

raw_af2_destress_data = pd.read_csv(raw_af2_destress_data_path)

# Defining a column which extracts the uniprot id from the design_name column
raw_af2_destress_data["uniprot_id"] = raw_af2_destress_data["design_name"].str.split("-").str[1]

print(raw_af2_destress_data["uniprot_id"])

raw_af2_destress_data["uniprot_id"].to_csv(raw_uniprot_data_path + "uniprot_id_list.csv", index=False)