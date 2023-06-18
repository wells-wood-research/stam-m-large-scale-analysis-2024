# 0. Loading in packages and defining custom functions---------------------------------------------------------
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 1. Defining variables--------------------------------------------------------------------------------------------------

# Defining file paths
uniprot_raw_data_path = "data/raw_data/uniprot/pdb/"
data_output_path = "data/processed_data/"

# # Creating a data frame to insert uniprot data into
# uniprot_data_df = pd.DataFrame(columns=["primary_accession", 
#                                         "uniProtkbId", 
#                                         "organism_scientific_name", 
#                                         "protein_name",
#                                         "pdb_id"])

# # Creating a data frame to insert go data into
# go_data_df = pd.DataFrame(columns=["primary_accession", 
#                                    "go_id", 
#                                    "go_col", 
#                                    "go_term"])


# 2. Reading json files--------------------------------------------------------------------------------------------------

# # Resolving the uniprot_raw_data_path that has been provided
# uniprot_raw_data_path = Path(uniprot_raw_data_path).resolve()

# # Getting a list of all the json files in the input path
# json_file_path_list = list(uniprot_raw_data_path.glob("*.json"))



# # Looping through them all and extracting required fields
# for json_file_path in json_file_path_list:

#     print(json_file_path)

#     # Reading in json file and extracting the results
#     with open(json_file_path, "r") as json_file:

#         # Loading in json file
#         json_file = json.load(json_file)

#         # Extracting results
#         uniprot_results = json_file["results"]

#         # Looping through the results and extracting the required fields
#         for n in range(0, len(uniprot_results)):
            
#             # Extracting the results for a specific uniprot id
#             uniprot_results_row = uniprot_results[n]["to"]

#             # PDB ID 
#             pdb_id = uniprot_results[n]["from"]

#             # Appending primary accession, uniProtkbId and organism scientific name
#             primary_accession = uniprot_results_row['primaryAccession']
#             uniProtkbId = uniprot_results_row['uniProtkbId']
#             organism_scientific_name = uniprot_results_row['organism']['scientificName']
            
#             # Extracting protein description
#             uniprot_results_row_proteindesc = uniprot_results_row['proteinDescription']
            
#             # If there is no recommended name then take the submission name
#             if "recommendedName" in uniprot_results_row_proteindesc.keys():
#                 protein_name = uniprot_results_row_proteindesc["recommendedName"]["fullName"]["value"]
#             elif "submissionNames" in uniprot_results_row_proteindesc.keys():
#                 protein_name = uniprot_results_row_proteindesc["submissionNames"][0]['fullName']["value"]


#             # # Extract gene name if there is one and set to None if not
#             # if "genes" in uniprot_results_row.keys():
#             #     # print(uniprot_results_row["genes"][0].keys())
#             #     if "geneName" in uniprot_results_row["genes"][0].keys():
#             #         gene_name_list.append(uniprot_results_row["genes"][0]["geneName"]["value"])
#             #     else:
#             #         gene_name_list.append("None")
#             # else:
#             #     gene_name_list.append("None")


#             # Inserting row into uniprot df
#             uniprot_data_df = uniprot_data_df.append({'primary_accession': primary_accession,
#                                                       'uniProtkbId': uniProtkbId,
#                                                       'organism_scientific_name': organism_scientific_name,
#                                                       'protein_name': protein_name,
#                                                       'pdb_id': pdb_id},
#                                                       ignore_index=True)

#             # # Extract GO terms
#             # uniprot_results_databases = uniprot_results[n]['to']['uniProtKBCrossReferences']
#             # for i in range(0, len(uniprot_results_databases)):
#             #     if uniprot_results_databases[i]['database'] == 'GO':

#             #         go_id = uniprot_results_databases[i]['id']
#             #         go_col = uniprot_results_databases[i]['properties'][0]['key']
#             #         go_term = uniprot_results_databases[i]['properties'][0]['value']

#             #         # Inserting row into go df
#             #         go_data_df = uniprot_data_df.append({'primary_accession': primary_accession,
#             #                                              'go_id': go_id,
#             #                                              'go_col': go_col,
#             #                                              'go_term': go_term},
#             #                                              ignore_index=True)


# uniprot_data_df = uniprot_data_df.drop_duplicates()
# uniprot_data_df.to_csv(data_output_path + "uniprot_data_df_pdb_ids.csv", index=False)

# # go_data_df = go_data_df.drop_duplicates()
# # go_data_df.to_csv(data_output_path + "go_data_df.csv", index=False)

# # uniprot_data_df_dupes = uniprot_data_df[uniprot_data_df.duplicated()]
# # uniprot_data_df_dupes.to_cs# Reading in destress pdb data
# # destress_data_pdb = pd.read_csv(destress_data_pdb_path)

# # # Extracting the pdb ids
# # pdb_id_list = list(destress_data_pdb["design_name"].str.replace("pdb", "").unique())

# # # Splitting the uniprot ids into batches of 100,000 
# # # (this is the max that the uniprot API can handle)
# # pdb_ids_list_batches = [pdb_id_list[i:i + 20000] for i in range(0, len(pdb_id_list), 20000)]

# # # Querying the uniprot database for the different 
# # # batches of uni prot ids
# # job_ids_list=[]
# # for pdb_ids_batch in pdb_ids_list_batches:
# #     job_id = query_uniprot_data(pdb_id_list=pdb_ids_batch)
# #     job_ids_list.append(job_id)


# # # Saving these job ids in a csv file
# # job_ids_df = pd.DataFrame(job_ids_list, columns=["job_id"])
# # job_ids_df.to_csv(data_output_path + "job_ids_df.csv", index=False)v(data_output_path + "uniprot_data_df_dupes.csv", index=False)

uniprot_data_df = pd.read_csv(data_output_path + "uniprot_data_df_af2_structures.csv")

uniprot_data_df["organism_scientific_name_af2"] = np.select(
    [
        uniprot_data_df["organism_scientific_name"].str.contains("Arabidopsis thaliana"),
        uniprot_data_df["organism_scientific_name"].str.contains("Caenorhabditis elegans"),
        uniprot_data_df["organism_scientific_name"].str.contains("Candida albicans"),
        uniprot_data_df["organism_scientific_name"].str.contains("Danio rerio"),
        uniprot_data_df["organism_scientific_name"].str.contains("Dictyostelium discoideum"),
        uniprot_data_df["organism_scientific_name"].str.contains("Drosophila melanogaster"),
        uniprot_data_df["organism_scientific_name"].str.contains("Escherichia coli"),
        uniprot_data_df["organism_scientific_name"].str.contains("Glycine max"),
        uniprot_data_df["organism_scientific_name"].str.contains("Homo sapiens"),
        uniprot_data_df["organism_scientific_name"].str.contains("Methanocaldococcus jannaschii"),
        uniprot_data_df["organism_scientific_name"].str.contains("Mus musculus"),
        uniprot_data_df["organism_scientific_name"].str.contains("Oryza sativa"),
        uniprot_data_df["organism_scientific_name"].str.contains("Rattus norvegicus"),
        uniprot_data_df["organism_scientific_name"].str.contains("Saccharomyces cerevisiae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Schizosaccharomyces pombe"),
        uniprot_data_df["organism_scientific_name"].str.contains("Zea mays"),
        uniprot_data_df["organism_scientific_name"].str.contains("Ajellomyces capsulatus"),
        uniprot_data_df["organism_scientific_name"].str.contains("Brugia malayi"),
        uniprot_data_df["organism_scientific_name"].str.contains("Campylobacter jejuni"),
        uniprot_data_df["organism_scientific_name"].str.contains("Cladophialophora carrionii"),
        uniprot_data_df["organism_scientific_name"].str.contains("Dracunculus medinensis"),
        uniprot_data_df["organism_scientific_name"].str.contains("Enterococcus faecium"),
        uniprot_data_df["organism_scientific_name"].str.contains("Fonsecaea pedrosoi"),
        uniprot_data_df["organism_scientific_name"].str.contains("Haemophilus influenzae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Helicobacter pylori"),
        uniprot_data_df["organism_scientific_name"].str.contains("Klebsiella pneumoniae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Leishmania infantum"),
        uniprot_data_df["organism_scientific_name"].str.contains("Madurella mycetomatis"),
        uniprot_data_df["organism_scientific_name"].str.contains("Mycobacterium leprae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Mycobacterium tuberculosis"),
        uniprot_data_df["organism_scientific_name"].str.contains("Mycobacterium ulcerans"),
        uniprot_data_df["organism_scientific_name"].str.contains("Neisseria gonorrhoeae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Nocardia brasiliensis"),
        uniprot_data_df["organism_scientific_name"].str.contains("Onchocerca volvulus"),
        uniprot_data_df["organism_scientific_name"].str.contains("Paracoccidioides lutzii"),
        uniprot_data_df["organism_scientific_name"].str.contains("Plasmodium falciparum"),
        uniprot_data_df["organism_scientific_name"].str.contains("Pseudomonas aeruginosa"),
        uniprot_data_df["organism_scientific_name"].str.contains("Salmonella typhimurium"),
        uniprot_data_df["organism_scientific_name"].str.contains("Schistosoma mansoni"),
        uniprot_data_df["organism_scientific_name"].str.contains("Shigella dysenteriae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Sporothrix schenckii"),
        uniprot_data_df["organism_scientific_name"].str.contains("Staphylococcus aureus"),
        uniprot_data_df["organism_scientific_name"].str.contains("Streptococcus pneumoniae"),
        uniprot_data_df["organism_scientific_name"].str.contains("Strongyloides stercoralis"),
        uniprot_data_df["organism_scientific_name"].str.contains("Trichuris trichiura"),
        uniprot_data_df["organism_scientific_name"].str.contains("Trypanosoma brucei"),
        uniprot_data_df["organism_scientific_name"].str.contains("Trypanosoma cruzi"),
        uniprot_data_df["organism_scientific_name"].str.contains("Wuchereria bancrofti"),

    ],
    ["Arabidopsis thaliana",
     "Caenorhabditis elegans",
     "Candida albicans",
     "Danio rerio",
     "Dictyostelium discoideum",
     "Drosophila melanogaster",
     "Escherichia coli",
     "Glycine max",
     "Homo sapiens",
     "Methanocaldococcus jannaschii",
     "Mus musculus",
     "Oryza sativa",
     "Rattus norvegicus",
     "Saccharomyces cerevisiae",
     "Schizosaccharomyces pombe",
     "Zea mays",
     "Ajellomyces capsulatus",
     "Brugia malayi",
     "Campylobacter jejuni",
     "Cladophialophora carrionii",
     "Dracunculus medinensis",
     "Enterococcus faecium",
     "Fonsecaea pedrosoi",
     "Haemophilus influenzae",
     "Helicobacter pylori",
     "Klebsiella pneumoniae",
     "Leishmania infantum",
     "Madurella mycetomatis",
     "Mycobacterium leprae",
     "Mycobacterium tuberculosis",
     "Mycobacterium ulcerans",
     "Neisseria gonorrhoeae",
     "Nocardia brasiliensis",
     "Onchocerca volvulus",
     "Paracoccidioides lutzii",
     "Plasmodium falciparum",
     "Pseudomonas aeruginosa",
     "Salmonella typhimurium",
     "Schistosoma mansoni",
     "Shigella dysenteriae",
     "Sporothrix schenckii",
     "Staphylococcus aureus",
     "Streptococcus pneumoniae",
     "Strongyloides stercoralis",
     "Trichuris trichiura",
     "Trypanosoma brucei",
     "Trypanosoma cruzi",
     "Wuchereria bancrofti",
     ],
    default="Other",
)

uniprot_data_df.to_csv(data_output_path + "processed_uniprot_data_af2.csv", index=True)
print(uniprot_data_df)



