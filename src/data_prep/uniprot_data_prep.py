# 0. Loading in packages and defining custom functions---------------------------------------------------------
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 1. Defining variables--------------------------------------------------------------------------------------------------

# Defining file paths
uniprot_raw_data_path = "data/raw_data/uniprot/"
data_output_path = "data/processed_data/"

# Initialising lists
primary_accession_list = []
uniProtkbId_list = []
organism_scientific_name_list = []
protein_name_list = []
gene_name_list = []

# 2. Reading json files--------------------------------------------------------------------------------------------------

# Resolving the uniprot_raw_data_path that has been provided
uniprot_raw_data_path = Path(uniprot_raw_data_path).resolve()

# Getting a list of all the json files in the input path
json_file_path_list = list(uniprot_raw_data_path.glob("*.json"))

# Looping through them all and extracting required fields
for json_file_path in json_file_path_list:

    # Reading in json file and extracting the results
    with open(json_file_path, "r") as json_file:

        # Loading in json file
        json_file = json.load(json_file)

        # Extracting results
        uniprot_results = json_file["results"]
        # Looping through the results and extracting the required fields
        for n in range(0, len(uniprot_results)):
            
            # Extracting the results for a specific uniprot id
            uniprot_results_row = uniprot_results[n]["to"]
            
            # Appending primary accession, uniProtkbId and organism scientific name
            primary_accession_list.append(uniprot_results_row['primaryAccession'])
            uniProtkbId_list.append(uniprot_results_row['uniProtkbId'])
            organism_scientific_name_list.append(uniprot_results_row['organism']['scientificName'])
            
            # Extracting protein description
            uniprot_results_row_proteindesc = uniprot_results_row['proteinDescription']
            
            # If there is no recommended name then take the submission name
            if "recommendedName" in uniprot_results_row_proteindesc.keys():
                protein_name_list.append(uniprot_results_row_proteindesc["recommendedName"]["fullName"]["value"])
            elif "submissionNames" in uniprot_results_row_proteindesc.keys():
                protein_name_list.append(uniprot_results_row_proteindesc["submissionNames"][0]['fullName']["value"])


            # Extract gene name if there is one and set to None if not
            if "genes" in uniprot_results_row.keys():
                # print(uniprot_results_row["genes"][0].keys())
                if "geneName" in uniprot_results_row["genes"][0].keys():
                    gene_name_list.append(uniprot_results_row["genes"][0]["geneName"]["value"])
                else:
                    gene_name_list.append("None")
            else:
                gene_name_list.append("None")

# Creating a dataframe with all of this data
uniprot_data_df = pd.DataFrame(list(zip(primary_accession_list, uniProtkbId_list, organism_scientific_name_list, protein_name_list, gene_name_list)), columns=["primary_accession", "uniProtkbId", "organism_scientific_name", "protein_name", "gene_name"])
uniprot_data_df["gene_name"] = np.where(uniprot_data_df["gene_name"] == "None", None, uniprot_data_df["gene_name"])
uniprot_data_df.to_csv(data_output_path + "uniprot_data_df.csv", index=False)