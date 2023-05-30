# 0. Loading in packages and defining custom functions---------------------------------------------------------
import numpy as np
import pandas as pd
import os
import subprocess
import json

# Defining a function to request data from the uniprotkb database 
# for a set of uniprot ids
def query_uniprot_data(pdb_id_list):

    # Create a string from the list of uniprot ids
    pdb_id_list_string = ",".join(pdb_id_list)
    
    # Defining the uniprot query curl command
    # to request data for a set of uniprot ids
    query_cmd = ["curl", 
            "--form", 
            'from="PDB"', 
            "--form", 
            'to="UniProtKB"', 
            "--form", 
            'ids=' + pdb_id_list_string, 
            "https://rest.uniprot.org/idmapping/run"]
    
    # Running this curl command
    uniprot_jobid_output = subprocess.run(query_cmd, capture_output=True)
    uniprot_jobid_output.check_returncode()
    
    # Extracting the job id
    job_id = uniprot_jobid_output.stdout.decode().split(":")[1].replace('"', '').replace("}", "")

    return job_id

# Defining a function to receive the uniprot results
# for specific job id
def receive_uniprot_data(job_id):

    # Defining the uniprot request results curl command
    # to receive the results for the job id
    request_results_cmd = ["curl",
                           "-s",
                           "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/" + job_id ]

    # Running this curl command
    uniprot_results_output = subprocess.run(request_results_cmd, capture_output=True)
    # print(uniprot_results_output)
    uniprot_results_output.check_returncode()
    
    # Extracting the result string
    result_string = uniprot_results_output.stdout.decode()

    return result_string


# 1. Defining variables--------------------------------------------------------------------------------------------------

# Defining file paths
destress_data_pdb_path = "data/raw_data/pdb/destress_data_pdb.csv"
data_output_path = "/home/michael/GitRepos/illuminating-protein-structural-universe/data/raw_data/uniprot/pdb/"

# 2. Querying the uniprot data and starting jobs--------------------------------------------------------------------------

# # Reading in destress pdb data
# destress_data_pdb = pd.read_csv(destress_data_pdb_path)

# # Extracting the pdb ids
# pdb_id_list = list(destress_data_pdb["design_name"].str.replace("pdb", "").unique())

# # Splitting the uniprot ids into batches of 100,000 
# # (this is the max that the uniprot API can handle)
# pdb_ids_list_batches = [pdb_id_list[i:i + 10000] for i in range(0, len(pdb_id_list), 10000)]

# # Querying the uniprot database for the different 
# # batches of uni prot ids
# job_ids_list=[]
# for pdb_ids_batch in pdb_ids_list_batches:
#     job_id = query_uniprot_data(pdb_id_list=pdb_ids_batch)
#     job_ids_list.append(job_id)


# # Saving these job ids in a csv file
# job_ids_df = pd.DataFrame(job_ids_list, columns=["job_id"])
# job_ids_df.to_csv(data_output_path + "job_ids_df.csv", index=False)


# 3. Downloading results from uniprot for a set of job ids------------------------------------------------------------------

# Read in job ids csv file
job_ids_df = pd.read_csv(data_output_path + "job_ids_df.csv")
job_ids_list = job_ids_df["job_id"].to_list()

# Receiving results for each of the job ids
for job_id in job_ids_list[0:4]:
    print(job_id)
    results = receive_uniprot_data(job_id=job_id)

    # Saving results as json file
    with open(data_output_path + job_id + '.json', 'w') as outfile:
        outfile.write(results)

    