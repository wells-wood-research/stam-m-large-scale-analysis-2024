# 0. Loading in packages and defining custom functions---------------------------------------------------------
import numpy as np
import pandas as pd
import os
import subprocess
import json

# Defining a function to request data from the uniprotkb database 
# for a set of uniprot ids
def query_uniprot_data(uniprot_id_list):

    # Create a string from the list of uniprot ids
    uniprot_id_list_string = ",".join(uniprot_id_list)
    
    # Defining the uniprot query curl command
    # to request data for a set of uniprot ids
    query_cmd = ["curl", 
                     "--form", 
                     'from="UniProtKB_AC-ID"', 
                     "--form", 
                     'to="UniProtKB"', 
                     "--form", 
                     'ids=' + uniprot_id_list_string, 
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
    uniprot_results_output.check_returncode()
    
    # Extracting the result string
    result_string = uniprot_results_output.stdout.decode()

    return result_string


# 1. Defining variables--------------------------------------------------------------------------------------------------

# Defining file paths
processed_af2_destress_data_path = "data/processed_data/processed_af2_destress_data.csv"
data_output_path = "data/raw_data/uniprot/"

# 2. Querying the uniprot data and starting jobs--------------------------------------------------------------------------

# Reading in processed af2 destress data
processed_af2_destress_data = pd.read_csv(processed_af2_destress_data_path)

# Extracting uniprot ids
uni_prot_ids_list = list(processed_af2_destress_data["uniprot_id"].unique())
print(len(uni_prot_ids_list))

# # Splitting the uniprot ids into batches of 100,000 
# # (this is the max that the uniprot API can handle)
# uni_prot_ids_list_batches = [uni_prot_ids_list[i:i + 10000] for i in range(0, len(uni_prot_ids_list), 10000)]

# # Querying the uniprot database for the different 
# # batches of uni prot ids
# job_ids_list=[]
# for uni_prot_ids_batch in uni_prot_ids_list_batches:
#     job_id = query_uniprot_data(uniprot_id_list=uni_prot_ids_batch)
#     job_ids_list.append(job_id)


# # Saving these job ids in a csv file
# job_ids_df = pd.DataFrame(job_ids_list, columns=["job_id"])
# job_ids_df.to_csv(data_output_path + "job_ids_df.csv", index=False)


# 3. Downloading results from uniprot for a set of job ids------------------------------------------------------------------

# # Read in job ids csv file
# job_ids_df = pd.read_csv(data_output_path + "job_ids_df.csv")
# job_ids_list = job_ids_df["job_id"].to_list()

# # Receiving results for each of the job ids
# for job_id in job_ids_list:
#     results = receive_uniprot_data(job_id=job_id)

#     # Saving results as json file
#     with open(data_output_path + job_id + '.json', 'w') as outfile:
#         outfile.write(results)

    

