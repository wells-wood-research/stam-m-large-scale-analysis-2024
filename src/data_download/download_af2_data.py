import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import subprocess
import multiprocessing as mp

NUM_WORKERS = 20

with open("Index of _pub_databases_alphafold_latest.html") as fp:
    soup = BeautifulSoup(fp, 'html.parser')

soup = soup.find_all("a")
soup_filtered = [x["href"] for x in soup if "UP" in x["href"]]

os.chdir("/scratch/alphafold_model_organisms/")


def download_tar_file(x):
    
    print(x)
    cmd = ["wget", x]
    subprocess.run(cmd)


# Initialising the mprocess pool and number of workers
with mp.Pool(processes=NUM_WORKERS) as process_pool:

    process_pool.map(download_tar_file, soup_filtered)
    



    