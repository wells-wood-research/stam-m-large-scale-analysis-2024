import subprocess
import os

os.chdir("/home/michael/")

# Define the command to run treedist
command = [
    "GitRepos/phylip-3.697/exe/treedist",
    "-f",
    "Downloads/phyliptree.phy",
]

try:
    completed_process = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
    )  # Timeout in seconds
    print(completed_process.stdout)
    print(completed_process.stderr)
except subprocess.TimeoutExpired:
    print("Command timed out.")
