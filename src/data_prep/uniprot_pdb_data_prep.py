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


# # 2. Reading json files--------------------------------------------------------------------------------------------------

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


#             # Inserting row into uniprot df
#             uniprot_data_df = uniprot_data_df.append({'primary_accession': primary_accession,
#                                                       'uniProtkbId': uniProtkbId,
#                                                       'organism_scientific_name': organism_scientific_name,
#                                                       'protein_name': protein_name,
#                                                       'pdb_id': pdb_id},
#                                                       ignore_index=True)


# uniprot_data_df = uniprot_data_df.drop_duplicates()
# uniprot_data_df.to_csv(data_output_path + "uniprot_data_df_pdb_ids.csv", index=False)


# 3. Processing uniprot data df-------------------------------------------------------------------

uniprot_data_df = pd.read_csv(data_output_path + "uniprot_data_df_pdb_ids.csv")
uniprot_data_df_filt = uniprot_data_df[["pdb_id", "organism_scientific_name"]]
uniprot_data_org_grouped = uniprot_data_df_filt.groupby('pdb_id')['organism_scientific_name'].apply(lambda x: ','.join(x))
uniprot_data_org_grouped = pd.DataFrame(uniprot_data_org_grouped, columns=["organism_scientific_name"])
uniprot_data_org_grouped["pdb_id"] = uniprot_data_org_grouped.index
uniprot_data_org_grouped.reset_index(drop=True, inplace=True)

def most_frequent(List):
    return max(set(List), key = List.count)


uniprot_data_org_grouped['organism_scientific_name_list'] = uniprot_data_org_grouped['organism_scientific_name'].str.split(",")
uniprot_data_org_grouped['organism_scientific_name_most_freq'] = uniprot_data_org_grouped['organism_scientific_name_list'].apply(lambda x: most_frequent(x))


uniprot_data_org_grouped["organism_scientific_name_pdb"] = np.select(
    [
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Arabidopsis thaliana"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Caenorhabditis elegans"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Candida albicans"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Danio rerio"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Dictyostelium discoideum"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Drosophila melanogaster"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Escherichia coli"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Glycine max"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Homo sapiens"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Methanocaldococcus jannaschii"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Mus musculus"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Oryza sativa"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Rattus norvegicus"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Saccharomyces cerevisiae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Schizosaccharomyces pombe"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Zea mays"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Ajellomyces capsulatus"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Brugia malayi"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Campylobacter jejuni"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Cladophialophora carrionii"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Dracunculus medinensis"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Enterococcus faecium"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Fonsecaea pedrosoi"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Haemophilus influenzae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Helicobacter pylori"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Klebsiella pneumoniae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Leishmania infantum"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Madurella mycetomatis"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Mycobacterium leprae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Mycobacterium tuberculosis"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Mycobacterium ulcerans"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Neisseria gonorrhoeae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Nocardia brasiliensis"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Onchocerca volvulus"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Paracoccidioides lutzii"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Plasmodium falciparum"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Pseudomonas aeruginosa"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Salmonella typhimurium"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Schistosoma mansoni"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Shigella dysenteriae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Sporothrix schenckii"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Staphylococcus aureus"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Streptococcus pneumoniae"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Strongyloides stercoralis"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Trichuris trichiura"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Trypanosoma brucei"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Trypanosoma cruzi"),
        uniprot_data_org_grouped["organism_scientific_name_most_freq"].str.contains("Wuchereria bancrofti"),

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


uniprot_data_org_grouped.to_csv(data_output_path + "processed_uniprot_data_pdb.csv", index=True)
print(uniprot_data_org_grouped)



	
	

