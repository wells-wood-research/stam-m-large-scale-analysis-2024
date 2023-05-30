# 0. Loading in packages and defining custom functions---------------------------------------------------------
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 1. Defining variables--------------------------------------------------------------------------------------------------

# Defining file paths
uniprot_raw_data_path = "data/raw_data/uniprot/pdb/"
data_output_path = "data/processed_data/"

# Creating a data frame to insert uniprot data into
uniprot_data_df = pd.DataFrame(columns=["primary_accession", 
                                        "uniProtkbId", 
                                        "organism_scientific_name", 
                                        "protein_name",
                                        "pdb_id"])

# # Creating a data frame to insert go data into
# go_data_df = pd.DataFrame(columns=["primary_accession", 
#                                    "go_id", 
#                                    "go_col", 
#                                    "go_term"])


# 2. Reading json files--------------------------------------------------------------------------------------------------

# Resolving the uniprot_raw_data_path that has been provided
uniprot_raw_data_path = Path(uniprot_raw_data_path).resolve()

# Getting a list of all the json files in the input path
json_file_path_list = list(uniprot_raw_data_path.glob("*.json"))



# Looping through them all and extracting required fields
for json_file_path in json_file_path_list:

    print(json_file_path)

    # Reading in json file and extracting the results
    with open(json_file_path, "r") as json_file:

        # Loading in json file
        json_file = json.load(json_file)

        # Extracting results
        uniprot_results = json_file["results"]
        print(uniprot_results)
        # Looping through the results and extracting the required fields
        for n in range(0, len(uniprot_results)):
            
            # Extracting the results for a specific uniprot id
            uniprot_results_row = uniprot_results[n]["to"]

            # PDB ID 
            pdb_id = uniprot_results[n]["from"]
            print(pdb_id)

            # Appending primary accession, uniProtkbId and organism scientific name
            primary_accession = uniprot_results_row['primaryAccession']
            uniProtkbId = uniprot_results_row['uniProtkbId']
            organism_scientific_name = uniprot_results_row['organism']['scientificName']
            
            # Extracting protein description
            uniprot_results_row_proteindesc = uniprot_results_row['proteinDescription']
            
            # If there is no recommended name then take the submission name
            if "recommendedName" in uniprot_results_row_proteindesc.keys():
                protein_name = uniprot_results_row_proteindesc["recommendedName"]["fullName"]["value"]
            elif "submissionNames" in uniprot_results_row_proteindesc.keys():
                protein_name = uniprot_results_row_proteindesc["submissionNames"][0]['fullName']["value"]


            # # Extract gene name if there is one and set to None if not
            # if "genes" in uniprot_results_row.keys():
            #     # print(uniprot_results_row["genes"][0].keys())
            #     if "geneName" in uniprot_results_row["genes"][0].keys():
            #         gene_name_list.append(uniprot_results_row["genes"][0]["geneName"]["value"])
            #     else:
            #         gene_name_list.append("None")
            # else:
            #     gene_name_list.append("None")


            # Inserting row into uniprot df
            uniprot_data_df = uniprot_data_df.append({'primary_accession': primary_accession,
                                                      'uniProtkbId': uniProtkbId,
                                                      'organism_scientific_name': organism_scientific_name,
                                                      'protein_name': protein_name,
                                                      'pdb_id': pdb_id},
                                                      ignore_index=True)

            # # Extract GO terms
            # uniprot_results_databases = uniprot_results[n]['to']['uniProtKBCrossReferences']
            # for i in range(0, len(uniprot_results_databases)):
            #     if uniprot_results_databases[i]['database'] == 'GO':

            #         go_id = uniprot_results_databases[i]['id']
            #         go_col = uniprot_results_databases[i]['properties'][0]['key']
            #         go_term = uniprot_results_databases[i]['properties'][0]['value']

            #         # Inserting row into go df
            #         go_data_df = uniprot_data_df.append({'primary_accession': primary_accession,
            #                                              'go_id': go_id,
            #                                              'go_col': go_col,
            #                                              'go_term': go_term},
            #                                              ignore_index=True)


uniprot_data_df = uniprot_data_df.drop_duplicates()
uniprot_data_df.to_csv(data_output_path + "uniprot_data_df_pdb_ids.csv", index=False)

# go_data_df = go_data_df.drop_duplicates()
# go_data_df.to_csv(data_output_path + "go_data_df.csv", index=False)

# uniprot_data_df_dupes = uniprot_data_df[uniprot_data_df.duplicated()]
# uniprot_data_df_dupes.to_cs# Reading in destress pdb data
# destress_data_pdb = pd.read_csv(destress_data_pdb_path)

# # Extracting the pdb ids
# pdb_id_list = list(destress_data_pdb["design_name"].str.replace("pdb", "").unique())

# # Splitting the uniprot ids into batches of 100,000 
# # (this is the max that the uniprot API can handle)
# pdb_ids_list_batches = [pdb_id_list[i:i + 20000] for i in range(0, len(pdb_id_list), 20000)]

# # Querying the uniprot database for the different 
# # batches of uni prot ids
# job_ids_list=[]
# for pdb_ids_batch in pdb_ids_list_batches:
#     job_id = query_uniprot_data(pdb_id_list=pdb_ids_batch)
#     job_ids_list.append(job_id)


# # Saving these job ids in a csv file
# job_ids_df = pd.DataFrame(job_ids_list, columns=["job_id"])
# job_ids_df.to_csv(data_output_path + "job_ids_df.csv", index=False)v(data_output_path + "uniprot_data_df_dupes.csv", index=False)